/******************************************************************************
 * Copyright (c) 2020, Bradley J Chambers (brad.chambers@gmail.com)
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following
 * conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided
 *       with the distribution.
 *     * Neither the name of Hobu, Inc. or Flaxen Geo Consulting nor the
 *       names of its contributors may be used to endorse or promote
 *       products derived from this software without specific prior
 *       written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 ****************************************************************************/

#include "SampledMongusFilter.hpp"

#include <pdal/KDIndex.hpp>
#include <pdal/private/MathUtils.hpp>
#include <pdal/util/ProgramArgs.hpp>

#include <Eigen/Dense>

#include <numeric>

namespace pdal
{
using namespace Dimension;
using namespace Eigen;

static PluginInfo const s_info{
    "filters.sampledmongus", "Sampled Mongus filter",
    "http://pdal.io/stages/filters.sampledmongus.html"};

CREATE_STATIC_STAGE(SampledMongusFilter, s_info)

std::string SampledMongusFilter::getName() const
{
    return s_info.name;
}

void SampledMongusFilter::addArgs(ProgramArgs& args)
{
    args.add("radius", "Radius", m_radius, 1.0);
    args.add("max_radius", "Max radius", m_maxRadius, 40.0);
    args.add("count", "Count", m_count, 10);
    args.add("thresh",
             "Points whose residual falls below the threshold are added to "
             "ground surface",
             m_thresh, 1.0);
    m_lambdaArg = &args.add("lambda", "Lambda for regularization", m_lambda);
    args.add("lambda_decay", "Decay rate of lambda", m_lambdaDecay, 0.1);
}

void SampledMongusFilter::addDimensions(PointLayoutPtr layout)
{
    layout->registerDim(Id::HeightAboveGround);
    layout->registerDim(Id::TopHat);
    layout->registerDim(Id::OpenErodeZ);
    layout->registerDim(Id::OpenDilateZ);
    layout->registerDim(Id::SurfaceEstimate);
    layout->registerDim(Id::Residual);
    layout->registerDim(Id::W);
    layout->registerDim(Id::NNDistance);
}

std::vector<PointIdList> SampledMongusFilter::buildScaleSpace(PointView& view)
{
    point_count_t min_pts = m_count;

    std::vector<PointIdList> ss;

    PointIdList samples;
    const KD2Index& index = view.build2dIndex();
    std::vector<int> keep(view.size(), 1);

    for (PointRef p : view)
    {
        if (keep[p.pointId()] == 0)
            continue;
        samples.push_back(p.pointId());
        PointIdList neighbors = index.radius(p, m_radius);
        for (PointId const& neighbor : neighbors)
        {
            if (neighbor == p.pointId())
                continue;
            keep[neighbor] = 0;
        }
    }

    ss.push_back(samples);

    log()->get(LogLevel::Debug)
        << "Sampled " << samples.size() << " points from " << view.size()
        << " at radius of " << m_radius << std::endl;

    double radius = m_radius;
    while ((radius <= m_maxRadius) && (samples.size() > min_pts))
    {
        radius *= 2.0;
        keep.clear();
        PointIdList samplesCopy(samples);
        samples.clear();
        for (PointId const& sample : samplesCopy)
        {
            if (keep[sample] == 0)
                continue;
            samples.push_back(sample);
            PointIdList neighbors = index.radius(sample, radius);
            for (PointId const& neighbor : neighbors)
            {
                if (neighbor == sample)
                    continue;
                keep[neighbor] = 0;
            }
        }

        if (samples.size() < m_count)
            return ss;

        ss.push_back(samples);

        log()->get(LogLevel::Debug)
            << "Sampled " << samples.size() << " points from " << view.size()
            << " at radius of " << radius << std::endl;
    }

    log()->get(LogLevel::Debug)
        << "Built scale space hierarchy with depth " << ss.size() << std::endl;
    return ss;
}

void SampledMongusFilter::interpolate(PointView& view, PointIdList ids,
                                      double radius)
{
    // Create a PointView consisting of only those points currently labeled as
    // ground.
    PointViewPtr gv = view.makeNew();
    for (PointRef p : view)
    {
        if (p.getFieldAs<uint8_t>(Id::Classification) == ClassLabel::Ground)
            gv->appendPoint(view, p.pointId());
    }
    const KD2Index& gIndex = gv->build2dIndex();

    point_count_t sum = 0;
    for (PointId const& id : ids)
    {
        PointRef p = view.point(id);
        auto CalcRbfValue = [](const VectorXd& xi, const VectorXd& xj) {
            double r = (xj - xi).norm();
            double value = r * r * std::log(r);
            return std::isnan(value) ? 0.0 : value;
        };

        PointIdList gIds(m_count);
        std::vector<double> gSqrDists(m_count);
        gIndex.knnSearch(p, m_count, &gIds, &gSqrDists);
        // PointIdList gIds = gIndex.radius(p, 8 * radius);
        sum += gIds.size();
        // m_count = gIds.size();
        MatrixXd X = MatrixXd::Zero(2, m_count);
        VectorXd y = VectorXd::Zero(m_count);
        VectorXd x = VectorXd::Zero(2);
        for (int i = 0; i < m_count; ++i)
        {
            X(0, i) = gv->getFieldAs<double>(Id::X, gIds[i]);
            X(1, i) = gv->getFieldAs<double>(Id::Y, gIds[i]);
            y(i) = gv->getFieldAs<double>(Id::Z, gIds[i]);
        }
        x(0) = p.getFieldAs<double>(Id::X);
        x(1) = p.getFieldAs<double>(Id::Y);
        MatrixXd Phi = MatrixXd::Zero(m_count, m_count);
        for (int i = 0; i < m_count; ++i)
        {
            for (int j = 0; j < m_count; ++j)
            {
                double value = CalcRbfValue(X.col(i), X.col(j));
                Phi(i, j) = Phi(j, i) = value;
            }
        }
        MatrixXd A = m_lambdaArg->set()
                         ? Phi.transpose() * Phi +
                               m_lambda * MatrixXd::Identity(m_count, m_count)
                         : Phi;
        VectorXd b = m_lambdaArg->set() ? Phi.transpose() * y : y;
        VectorXd w = Eigen::FullPivLU<MatrixXd>(A).solve(b);

        double val(0.0);
        for (int i = 0; i < m_count; ++i)
        {
            val += w(i) * CalcRbfValue(x, X.col(i));
        }
        p.setField(Id::SurfaceEstimate, val);
        p.setField(Id::Residual, p.getFieldAs<double>(Id::Z) - val);
        p.setField(Id::NNDistance, std::sqrt(gSqrDists.back()));
    }
    log()->get(LogLevel::Debug)
        << "ground contains " << gv->size() << " control points\n";
    log()->get(LogLevel::Debug)
        << "found average of " << sum / ids.size() << " control points within "
        << 8 * radius * m_maxrange << std::endl;
}

void SampledMongusFilter::tophat(PointView& view, PointIdList ids,
                                 double radius)
{
    PointViewPtr candView = view.makeNew();
    for (PointId const& id : ids)
        candView->appendPoint(view, id);
    const KD2Index& candIndex = candView->build2dIndex();

    // apply white top hat transform to residuals
    // erosion then dilation at 2*l (assume that means radius at current level)
    typedef std::map<PointId, PointIdList> NeighborMap;
    typedef std::map<PointId, double> ValueMap;
    NeighborMap nm;
    ValueMap vm;

    point_count_t sum = 0;
    for (PointRef p : *candView)
    {
        double radius = p.getFieldAs<double>(Id::NNDistance);
        // log()->get(LogLevel::Debug) << 2 * radius << std::endl;
        PointIdList cIds = candIndex.radius(p, 2 * radius);
        sum += cIds.size();
        nm[p.pointId()] = cIds;
        std::vector<double> z(cIds.size());
        for (size_t i = 0; i < cIds.size(); ++i)
            z[i] = candView->getFieldAs<double>(Id::Residual, cIds[i]);
        double val = *std::min_element(z.begin(), z.end());
        p.setField(Id::OpenErodeZ, val);
    }

    log()->get(LogLevel::Debug)
        << "found average of " << sum / candView->size() << " neighbors within "
        << 2 * radius * m_maxrange << std::endl;

    for (PointRef p : *candView)
    {
        PointIdList cIds = nm[p.pointId()];
        std::vector<double> z(cIds.size());
        for (size_t i = 0; i < cIds.size(); ++i)
            z[i] = candView->getFieldAs<double>(Id::OpenErodeZ, cIds[i]);
        double vm2 = *std::max_element(z.begin(), z.end());
        p.setField(Id::OpenDilateZ, vm2);
        p.setField(Id::TopHat, (p.getFieldAs<double>(Id::Residual) -
                                p.getFieldAs<double>(Id::OpenDilateZ)));
    }

    for (PointRef p : *candView)
    {
        PointIdList cIds = nm[p.pointId()];
        double M1, M2;
        M1 = M2 = 0.0;
        point_count_t cnt = 0;
        for (size_t i = 0; i < cIds.size(); ++i)
        {
            PointRef p = view.point(cIds[i]);
            point_count_t n(cnt++);
            double delta = p.getFieldAs<double>(Id::TopHat) - M1;
            double delta_n = delta / cnt;
            M1 += delta_n;
            M2 += delta * delta_n * n;
        }
        p.setField(Id::W, M1 + m_thresh * std::sqrt(M2 / (cnt - 1.0)));
    }
}

void SampledMongusFilter::classifyPoints(PointView& view, PointIdList ids)
{
    point_count_t kept = 0;
    point_count_t g, ng;
    g = ng = 0;
    for (PointId const& id : ids)
    {
        PointRef p = view.point(id);
        if (p.getFieldAs<double>(Id::TopHat) < p.getFieldAs<double>(Id::W))
        {
            if (p.getFieldAs<uint8_t>(Id::Classification) == ClassLabel::Ground)
                g++;
            else
                ng++;
            p.setField(Id::Classification, ClassLabel::Ground);
            kept++;
        }
    }
    log()->get(LogLevel::Debug) << "added " << ng << " control points from "
                                << ids.size() << " candidates" << std::endl;
}

void SampledMongusFilter::filter(PointView& view)
{
    // Return if the input PointView has no points.
    if (!view.size())
        return;

    // We want to visit the points in ascending elevation order.
    // I'd rather reorder the indices and not the points themselves, but
    // skipping for now...
    // std::vector<PointId> indices(view->size());
    // std::iota(indices.begin(), indices.end(), 0);
    auto cmp = [this](const PointRef& p1, const PointRef& p2) {
        return p1.compare(Id::Z, p2);
    };
    std::stable_sort(view.begin(), view.end(), cmp);

    // Build scale space decomposition, starting with base radius, which should
    // be closely aligned with density of data and target resolution of derived
    // products (like bare earth DEMs). After sampling at target radius,
    // continue to sample recursively at progressively larger radii (2x at each
    // iteration).
    std::vector<PointIdList> ss = buildScaleSpace(view);

    // Shift and scale the point cloud to fit within [-1,1] in each of XYZ.
    view.calculateBounds(m_bounds);
    double xrange = m_bounds.maxx - m_bounds.minx;
    double yrange = m_bounds.maxy - m_bounds.miny;
    double zrange = m_bounds.maxz - m_bounds.minz;
    m_maxrange = 2 * std::max(xrange, std::max(yrange, zrange));
    PointIdList ids(view.size());
    std::iota(ids.begin(), ids.end(), 0);
    Vector3d centroid = math::computeCentroid(view, ids);
    for (PointRef p : view)
    {
        double x = p.getFieldAs<double>(Id::X) - centroid.x();
        double y = p.getFieldAs<double>(Id::Y) - centroid.y();
        double z = p.getFieldAs<double>(Id::Z) - centroid.z();
        p.setField(Id::X, x / m_maxrange);
        p.setField(Id::Y, y / m_maxrange);
        p.setField(Id::Z, z / m_maxrange);
    }

    // Start by relabeling all points as Unclassified.
    for (PointRef p : view)
        p.setField(Id::Classification, ClassLabel::Unclassified);

    // The last element in the scale space decomposition are the minimum Z
    // values at the coarsest level of the hierarchy. We consider them as our
    // putative ground returns and label them accordingly.
    PointIdList seeds = ss.back();
    ss.pop_back();
    for (PointId const& id : seeds)
        view.setField(Id::Classification, id, ClassLabel::Ground);

    // As long as we still have levels in the scale space to traverse, we
    // repeat the following filtering steps.
    while (ss.size())
    {
        // Get the candidate ground returns at the next level of the hierarchy.
        PointIdList candidates = ss.back();
        ss.pop_back();

        // The radius we use for interpolation and morphological operations is
        // determined by the current level of the hierarchy (scaled by maxrange
        // now that the data has been scaled).
        double radius = std::pow(2.0, ss.size()) / m_maxrange;

        // Compute an interpolated value for each candidate ground return.
        interpolate(view, candidates, radius);

        // Compute the residual, top hat transform, and local threshold value
        // for each candidate ground return.
        tophat(view, candidates, radius);

        // Classify points according to the computed values.
        classifyPoints(view, candidates);
    }

    // Undo the shifting/scaling of the data to place it back in the original
    // coordinate space.
    for (PointRef p : view)
    {
        double x = p.getFieldAs<double>(Id::X) * m_maxrange;
        double y = p.getFieldAs<double>(Id::Y) * m_maxrange;
        double z = p.getFieldAs<double>(Id::Z) * m_maxrange;
        p.setField(Id::X, x + centroid.x());
        p.setField(Id::Y, y + centroid.y());
        p.setField(Id::Z, z + centroid.z());
    }
}

} // namespace pdal
