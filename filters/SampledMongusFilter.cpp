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
    args.add("count", "Count", m_count, 10);
    args.add("max_iters", "Maximum number of iterations", m_maxiters, 4);
    args.add("thresh",
             "Points whose residual falls below the threshold are added to "
             "ground surface",
             m_thresh, 1.0);
    args.add("radius_decay", "Decay rate of radius", m_radDecay, 0.5);
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
}

std::vector<PointIdList> SampledMongusFilter::buildScaleSpace(PointView& view)
{
    // things that should be constant or parameters, we'll hardcode here for now
    double target_radius = 1.0;
    double max_radius = 40.0;
    point_count_t min_pts = 3;

    std::vector<PointIdList> ss;

    PointIdList samples;
    const KD2Index& index = view.build2dIndex();
    std::vector<int> keep(view.size(), 1);

    for (PointRef p : view)
    {
        if (keep[p.pointId()] == 0)
            continue;
        samples.push_back(p.pointId());
        PointIdList neighbors = index.radius(p, target_radius);
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
        << " at radius of " << target_radius << std::endl;

    double radius = target_radius;
    while ((radius <= max_radius) && (samples.size() > min_pts))
    {
        radius *= 2.0;
        //PointViewPtr sampledView = view.makeNew();
        //for (PointId const& sample : samples)
        //    sampledView->appendPoint(view, sample);
        //const KD2Index& kdi = sampledView->build2dIndex();
        //std::vector<int> sampledKeep(sampledView->size(), 1);
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

        for (PointRef p : view)
        {
            if (keep[p.pointId()] == 0)
                continue;
            samples.push_back(p.pointId());
            PointIdList neighbors = index.radius(p, radius);
            for (PointId const& neighbor : neighbors)
            {
                if (neighbor == p.pointId())
                    continue;
                keep[neighbor] = 0;
            }
        }

        ss.push_back(samples);

        log()->get(LogLevel::Debug)
            << "Sampled " << samples.size() << " points from "
            << view.size() << " at radius of " << radius << std::endl;
    }

    log()->get(LogLevel::Debug) << ss.size() << std::endl;
    return ss;
}

PointIdList SampledMongusFilter::sample(PointView& view)
{
    PointIdList emptyids;
    PointIdList samples = sample(view, emptyids);
    //    for (PointId const& id : samples)
    //        view.setField(Id::Classification, id, ClassLabel::Ground);
    return samples;
}

PointIdList SampledMongusFilter::sample(PointView& view, PointIdList ids)
{
    PointIdList samples;
    const KD2Index& index = view.build2dIndex();
    std::vector<int> keep(view.size(), 1);

    for (PointId const& id : ids)
    {
        if (keep[id] == 0)
            continue;
        PointIdList neighbors = index.radius(id, m_radius);
        for (PointId const& neighbor : neighbors)
        {
            if (neighbor == id)
                continue;
            keep[neighbor] = 0;
        }
    }

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

    log()->get(LogLevel::Debug)
        << "Sampled " << std::accumulate(keep.begin(), keep.end(), 0)
        << " points from " << view.size() << " at radius of "
        << m_radius * m_maxrange << " (" << samples.size() << " > "
        << ids.size() << ")" << std::endl;

    return samples;
}

void SampledMongusFilter::interpolate(PointViewPtr candView, PointViewPtr gView)
{
    const KD2Index& gIndex = gView->build2dIndex();
    const KD2Index& candIndex = candView->build2dIndex();

    for (PointRef p : *candView)
    {
        auto CalcRbfValue = [](const VectorXd& xi, const VectorXd& xj) {
            double r = (xj - xi).norm();
            double value = r * r * std::log(r);
            return std::isnan(value) ? 0.0 : value;
        };
        PointIdList gIds = gIndex.neighbors(p, m_count);
        MatrixXd X = MatrixXd::Zero(2, m_count);
        VectorXd y = VectorXd::Zero(m_count);
        VectorXd x = VectorXd::Zero(2);
        for (int i = 0; i < m_count; ++i)
        {
            X(0, i) = gView->getFieldAs<double>(Id::X, gIds[i]);
            X(1, i) = gView->getFieldAs<double>(Id::Y, gIds[i]);
            y(i) = gView->getFieldAs<double>(Id::Z, gIds[i]);
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
        p.setField(Id::Residual, p.getFieldAs<double>(Id::Z));
    }
}

void SampledMongusFilter::tophat(PointViewPtr candView)
{
    const KD2Index& candIndex = candView->build2dIndex();

    // apply white top hat transform to residuals
    // erosion then dilation at 2*l (assume that means radius at current level)
    typedef std::map<PointId, PointIdList> NeighborMap;
    typedef std::map<PointId, double> ValueMap;
    NeighborMap nm;
    ValueMap vm;

    for (PointRef p : *candView)
    {
        PointIdList cIds = candIndex.radius(p, 8 * m_radius);
        nm[p.pointId()] = cIds;
        std::vector<double> z(cIds.size());
        for (size_t i = 0; i < cIds.size(); ++i)
            z[i] = candView->getFieldAs<double>(Id::Residual, cIds[i]);
        double val = *std::min_element(z.begin(), z.end());
        p.setField(Id::OpenErodeZ, val);
    }

    for (PointRef p : *candView)
    {
        PointIdList cIds = nm[p.pointId()];
        std::vector<double> z(cIds.size());
        for (size_t i = 0; i < cIds.size(); ++i)
            z[i] = candView->getFieldAs<double>(Id::OpenErodeZ, cIds[i]);
        double vm2 = *std::max_element(z.begin(), z.end());
        p.setField(Id::OpenDilateZ, vm2);
    }
}

PointIdList SampledMongusFilter::foo(PointView& view, PointIdList ids)
{
    // build view from ids and index it
    PointViewPtr gView = view.makeNew();
    for (PointId const& id : ids)
        gView->appendPoint(view, id);

    PointIdList candidates = sample(view, ids);
    PointViewPtr candView = view.makeNew();
    for (PointId const& candidate : candidates)
        candView->appendPoint(view, candidate);
    const KD2Index& candIndex = candView->build2dIndex();

    interpolate(candView, gView);

    tophat(candView);

    double M1, M2;
    M1 = M2 = 0.0;
    point_count_t cnt = 0;
    for (PointRef p : *candView)
    {
        point_count_t n(cnt++);
        double delta = (p.getFieldAs<double>(Id::Residual) -
                        p.getFieldAs<double>(Id::OpenDilateZ)) -
                       M1;
        double delta_n = delta / cnt;
        M1 += delta_n;
        M2 += delta * delta_n * n;
    }

    log()->get(LogLevel::Debug)
        << M1 * m_maxrange << " + " << m_thresh << " * "
        << std::sqrt(M2 / (cnt - 1.0)) * m_maxrange << " = "
        << (M1 + m_thresh * std::sqrt(M2 / (cnt - 1.0))) * m_maxrange
        << std::endl;

    PointIdList kept(ids);
    for (PointRef p : *candView)
    {
        if ((p.getFieldAs<double>(Id::Residual) -
             p.getFieldAs<double>(Id::OpenDilateZ)) <
            (M1 + m_thresh * std::sqrt(M2 / (cnt - 1.0))))
        {
            kept.push_back(candidates[p.pointId()]);
            // p.setField(Id::Classification, ClassLabel::Ground);
        }
        // std::cerr << p.pointId() << ", " << candidates[p.pointId()] << ", "
        // << p.getFieldAs<double>(Id::Residual) << ", " <<
        // p.getFieldAs<double>(Id::OpenDilateZ) << std::endl;
        p.setField(Id::TopHat, (p.getFieldAs<double>(Id::Residual) -
                                p.getFieldAs<double>(Id::OpenDilateZ)));
    }
    log()->get(LogLevel::Debug)
        << "Keep " << kept.size() << " of " << candidates.size() << "+"
        << ids.size() << "=" << candidates.size() + ids.size() << std::endl;

    return kept;
}

void SampledMongusFilter::bar(PointView& view, PointIdList ids)
{
    // build view from ids and index it
    PointViewPtr gView = view.makeNew();
    for (PointId const& id : ids)
        gView->appendPoint(view, id);
    const KD2Index& gIndex = gView->build2dIndex();

    // PointIdList candidates = sample(view, radius);
    typedef std::map<PointId, double> ValueMap;
    ValueMap residuals;
    // PointViewPtr candView = view.makeNew();
    // for (PointId const& candidate : candidates)
    //	    candView->appendPoint(view, candidate);
    // const KD2Index& candIndex = candView->build2dIndex();
    const KD2Index& index = view.build2dIndex();
    for (PointRef p : view)
    {
        auto CalcRbfValue = [](const VectorXd& xi, const VectorXd& xj) {
            double r = (xj - xi).norm();
            double value = r * r * std::log(r);
            // std::cerr << "r: " << r << ", rbf: " << value << std::endl;
            return std::isnan(value) ? 0.0 : value;
        };
        PointIdList gIds = gIndex.neighbors(p, m_count);
        MatrixXd X = MatrixXd::Zero(2, m_count);
        VectorXd y = VectorXd::Zero(m_count);
        VectorXd x = VectorXd::Zero(2);
        for (int i = 0; i < m_count; ++i)
        {
            X(0, i) = gView->getFieldAs<double>(Id::X, gIds[i]);
            X(1, i) = gView->getFieldAs<double>(Id::Y, gIds[i]);
            y(i) = gView->getFieldAs<double>(Id::Z, gIds[i]);
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
        VectorXd w = Eigen::PartialPivLU<MatrixXd>(A).solve(b);

        double val(0.0);
        for (int i = 0; i < m_count; ++i)
        {
            val += w(i) * CalcRbfValue(x, X.col(i));
        }
        // std::cerr << "neighbors: " << y.transpose() * m_maxrange<< std::endl;
        // std::cerr << "interpolated value: " << val * m_maxrange << std::endl;
        // std::cerr << "xyz: " << x.transpose() * m_maxrange << "  " <<
        // p.getFieldAs<double>(Id::Z) * m_maxrange << std::endl;
        residuals[p.pointId()] = p.getFieldAs<double>(Id::Z) - val;
        // std::cerr << "residual: " << residuals[p.pointId()] << " (" <<
        // residuals[p.pointId()]*m_maxrange << ")" << std::endl;
        p.setField(Id::HeightAboveGround, residuals[p.pointId()] * m_maxrange);
        p.setField(Id::Z, val);
    }

    // apply white top hat transform to residuals
    // erosion then dilation at 2*l (assume that means radius at current level)
    typedef std::map<PointId, PointIdList> NeighborMap;
    NeighborMap nm;
    ValueMap vm;

    for (PointRef p : view)
    {
        PointIdList cIds = index.radius(p, 2 * m_radius);
        nm[p.pointId()] = cIds;
        std::vector<double> z(cIds.size());
        for (size_t i = 0; i < cIds.size(); ++i)
            z[i] = residuals[cIds[i]];
        vm[p.pointId()] = *std::min_element(z.begin(), z.end());
    }

    for (PointRef p : view)
    {
        // could/should clear z first, but it will be overwritten
        PointIdList cIds = nm[p.pointId()];
        std::vector<double> z(cIds.size());
        for (size_t i = 0; i < cIds.size(); ++i)
            z[i] = vm[cIds[i]];
        double vm2 = *std::max_element(z.begin(), z.end());

        double M1, M2;
        M1 = M2 = 0.0;
        point_count_t cnt = 0;
        for (size_t i = 0; i < cIds.size(); ++i)
        {
            point_count_t n(cnt++);
            double delta = (residuals[cIds[i]] - vm2) - M1;
            double delta_n = delta / cnt;
            M1 += delta_n;
            M2 += delta * delta_n * n;
        }

        log()->get(LogLevel::Debug) << p.pointId() << ": " << M1 << ", " << M2
                                    << ", " << cnt << std::endl;
        log()->get(LogLevel::Debug)
            << M1 * m_maxrange << ", "
            << std::sqrt(M2 / (cnt - 1.0)) * m_maxrange << std::endl;

        if ((residuals[p.pointId()] - vm2) <
            (M1 + 3 * std::sqrt(M2 / (cnt - 1.0))))
            p.setField(Id::Classification, ClassLabel::Ground);
        p.setField(Id::TopHat, residuals[p.pointId()] - vm2);
    }
}

void SampledMongusFilter::baz(PointView& view, PointIdList ids)
{
    // build view from ids and index it
    PointViewPtr gView = view.makeNew();
    for (PointId const& id : ids)
        gView->appendPoint(view, id);
    const KD2Index& gIndex = gView->build2dIndex();

    for (PointRef p : view)
    {
        auto CalcRbfValue = [](const VectorXd& xi, const VectorXd& xj) {
            double r = (xj - xi).norm();
            double value = r * r * std::log(r);
            // std::cerr << "r: " << r << ", rbf: " << value << std::endl;
            return std::isnan(value) ? 0.0 : value;
        };
        PointIdList gIds = gIndex.neighbors(p, m_count);
        MatrixXd X = MatrixXd::Zero(2, m_count);
        VectorXd y = VectorXd::Zero(m_count);
        VectorXd x = VectorXd::Zero(2);
        for (int i = 0; i < m_count; ++i)
        {
            X(0, i) = gView->getFieldAs<double>(Id::X, gIds[i]);
            X(1, i) = gView->getFieldAs<double>(Id::Y, gIds[i]);
            y(i) = gView->getFieldAs<double>(Id::Z, gIds[i]);
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
        VectorXd w = Eigen::PartialPivLU<MatrixXd>(A).solve(b);

        double val(0.0);
        for (int i = 0; i < m_count; ++i)
        {
            val += w(i) * CalcRbfValue(x, X.col(i));
        }
        p.setField(Id::HeightAboveGround,
                   m_maxrange * (p.getFieldAs<double>(Id::Z) - val));
        p.setField(Id::Z, val);
    }
}

void SampledMongusFilter::filter(PointView& inView)
{
    // Return empty PointViewSet if the input PointView has no points.
    // Otherwise, make a new output PointView.
    if (!inView.size())
        return;

    // I'd rather reorder the indices and not the points themselves, but
    // skipping for now...
    // std::vector<PointId> indices(inView->size());
    // std::iota(indices.begin(), indices.end(), 0);
    auto cmp = [this](const PointRef& p1, const PointRef& p2) {
        return p1.compare(Id::Z, p2);
    };
    std::stable_sort(inView.begin(), inView.end(), cmp);

    // before we do anything else, we can create our poisson sampled scale space
    // representation by first sampling at the target radius, then resampling
    // the samples at increasingly larger radii (e.g., 2x at each iteration)
    // until max radius reached, or there are too few samples (say at least
    // three are required) this could be a map of iteration/radius to
    // PointIdList or iteration/radius to PointView, not sure which is easier
    // better, or maybe we just keep a temp copy at each iteration because we
    // will want to unroll the scale space, i think we do want to store the map.
    std::vector<PointIdList> ss = buildScaleSpace(inView);

    // once we reach the top level (max radius or min points), we mark these as
    // our first set of ground returns and immediately proceed to the next level
    PointIdList seeds = ss.back();
    std::cerr << seeds.size() << std::endl;
    ss.pop_back();
    for (PointId const& id : seeds)
	    inView.setField(Id::Classification, id, ClassLabel::Ground);

    PointIdList seeds2 = ss.back();
    std::cerr << seeds2.size() << std::endl;
    ss.pop_back();
    for (PointId const& id : seeds2)
	    inView.setField(Id::Classification, id, ClassLabel::LowVegetation);


    // somewhere in here we will want to scale to unit cube, probably after we
    // have our PointIdList maps set now we can call our old "foo", which in
    // reality is
    //   - interpolate current samples using previous surface estimate
    //   - compute top hat of residuals
    //   - compute threshold (neighborhood? global?)
    //   - apply threshold and keep only good samples, label as ground
    // afterwards, we can compute one final residual for HAG, and perhaps even
    // label as ground those that are close enough

    /*
    inView.calculateBounds(m_bounds);
    double xrange = m_bounds.maxx - m_bounds.minx;
    double yrange = m_bounds.maxy - m_bounds.miny;
    double zrange = m_bounds.maxz - m_bounds.minz;
    m_maxrange = 2 * std::max(xrange, std::max(yrange, zrange));

    m_radius /= m_maxrange;

    PointIdList ids(inView.size());
    std::iota(ids.begin(), ids.end(), 0);
    Vector3d centroid = math::computeCentroid(inView, ids);

    for (PointRef p : inView)
    {
        double x = p.getFieldAs<double>(Id::X) - centroid.x();
        double y = p.getFieldAs<double>(Id::Y) - centroid.y();
        double z = p.getFieldAs<double>(Id::Z) - centroid.z();
        p.setField(Id::X, x / m_maxrange);
        p.setField(Id::Y, y / m_maxrange);
        p.setField(Id::Z, z / m_maxrange);
    }

    for (PointRef p : inView)
        p.setField(Id::Classification, ClassLabel::Unclassified);

    log()->get(LogLevel::Info) << "Finding seed points\n";
    PointIdList samples = sample(inView);

    PointIdList groundsamples(samples);
    for (int iter = 0; iter < m_maxiters; ++iter)
    {
        m_radius *= 0.5;
        PointIdList newSamples = foo(inView, groundsamples);
        groundsamples.swap(newSamples);
        std::cerr << groundsamples.size() << std::endl;
    }

    // need a final pass that compares ALL points to the TPS and records HAG
    // bar(inView, groundsamples);
    // baz(inView, groundsamples);

    // instead of finalizing, for now maybe just write out the current control
    // points
    for (PointId const& id : groundsamples)
        inView.setField(Id::Classification, id, ClassLabel::Ground);

    for (PointRef p : inView)
    {
        double x = p.getFieldAs<double>(Id::X) * m_maxrange;
        double y = p.getFieldAs<double>(Id::Y) * m_maxrange;
        double z = p.getFieldAs<double>(Id::Z) * m_maxrange;
        p.setField(Id::X, x + centroid.x());
        p.setField(Id::Y, y + centroid.y());
        p.setField(Id::Z, z + centroid.z());
    }
    */
}

} // namespace pdal
