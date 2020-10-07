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

#include "MinSampleFilter.hpp"

#include <pdal/KDIndex.hpp>
#include <pdal/private/MathUtils.hpp>
#include <pdal/util/ProgramArgs.hpp>

#include <Eigen/Dense>

#include <numeric>

namespace pdal
{
using namespace Dimension;
using namespace Eigen;

static PluginInfo const s_info{"filters.minsample",
                               "Minimum Subsampling filter",
                               "http://pdal.io/stages/filters.minsample.html"};

CREATE_STATIC_STAGE(MinSampleFilter, s_info)

std::string MinSampleFilter::getName() const
{
    return s_info.name;
}

void MinSampleFilter::addArgs(ProgramArgs& args)
{
    args.add("radius", "Radius", m_radius, 1.0);
    args.add("count", "Count", m_count, 10);
    args.add("max_iters", "Maximum number of iterations", m_maxiters, 4);
    args.add("thresh",
             "Points whose residual falls below the threshold are added to "
             "ground surface",
             m_thresh, 1.0);
    args.add("radius_decay", "Decay rate of radius", m_radDecay, 0.5);
    args.add("thresh_decay", "Decay rate of thresh", m_threshDecay, 0.5);
    m_lambdaArg = &args.add("lambda", "Lambda for regularization", m_lambda);
    args.add("lambda_decay", "Decay rate of lambda", m_lambdaDecay, 0.1);
}

void MinSampleFilter::addDimensions(PointLayoutPtr layout)
{
    layout->registerDim(Id::HeightAboveGround);
}

PointViewPtr MinSampleFilter::maskNeighbors(PointView& view,
                                            const KD2Index& index)
{
    // Iterate through the view, marking selected points as ground and
    // masking neighbors within m_radius. These masked neighbors are marked
    // as unclassified and are removed from further consideration for this
    // iteration. This is simply Poisson disk throwing, but because the
    // point view is sorted by increasing elevation (outside the function),
    // it serves as a crude initial estimate of ground.

    // Create the ground PointViewPtr, which will be returned at the end of the
    // function.
    PointViewPtr gView = view.makeNew();
    for (PointRef p : view)
    {
        // If a point is masked, it is forever masked, and cannot be part of the
        // sampled ground surface. Otherwise, the current index is appended to
        // the output PointView.
        if (p.getFieldAs<uint8_t>(Id::Classification) ==
            ClassLabel::Unclassified)
            continue;

        // Both classify the current point as ground and also add to the output
        // PointView.
        p.setField(Id::Classification, ClassLabel::Ground);
        gView->appendPoint(view, p.pointId());

        // We now proceed to mask all neighbors within m_radius of the kept
        // point.
        PointIdList ids = index.radius(p, m_radius);
        for (PointId const& j : ids)
        {
            if (j == p.pointId())
                continue;
            view.setField(Id::Classification, j, ClassLabel::Unclassified);
        }
    }

    point_count_t numGround = 0;
    point_count_t numUnclass = 0;
    point_count_t numNeverClass = 0;
    for (PointRef p : view)
    {
        uint8_t cls = p.getFieldAs<uint8_t>(Id::Classification);
        if (cls == ClassLabel::Ground)
            ++numGround;
        else if (cls == ClassLabel::Unclassified)
            ++numUnclass;
        else if (cls == ClassLabel::CreatedNeverClassified)
            ++numNeverClass;
        else
            log()->get(LogLevel::Error) << "Shouldn't happen\n";
    }

    log()->get(LogLevel::Debug)
        << "Radius: " << m_radius << ", thresh: " << m_thresh
        << ", lambda: " << m_lambda << ", ground: " << numGround
        << ", unclass: " << numUnclass << ", never: " << numNeverClass
        << ", total: " << view.size() << std::endl;

    return gView;
}

void MinSampleFilter::maskGroundNeighbors(PointView& view,
                                          const KD2Index& index)
{
    for (PointRef p : view)
    {
        if (p.getFieldAs<uint8_t>(Id::Classification) ==
            ClassLabel::Unclassified)
            p.setField(Id::Classification, ClassLabel::CreatedNeverClassified);
    }
    for (PointRef p : view)
    {
        if (p.getFieldAs<uint8_t>(Id::Classification) == ClassLabel::Ground)
        {
            PointIdList ids = index.radius(p, m_radius);

            // We now proceed to mask all neighbors within m_radius of the kept
            // point.
            for (PointId const& j : ids)
            {
                if (j == p.pointId())
                    continue;
                if (view.getFieldAs<uint8_t>(Id::Classification, j) ==
                    ClassLabel::Ground)
                    log()->get(LogLevel::Error) << "Shouldn't happen\n";
                view.setField(Id::Classification, j, ClassLabel::Unclassified);
            }
        }
    }

    point_count_t numGround = 0;
    point_count_t numUnclass = 0;
    point_count_t numNeverClass = 0;
    for (PointRef p : view)
    {
        uint8_t cls = p.getFieldAs<uint8_t>(Id::Classification);
        if (cls == ClassLabel::Ground)
            ++numGround;
        else if (cls == ClassLabel::Unclassified)
            ++numUnclass;
        else if (cls == ClassLabel::CreatedNeverClassified)
            ++numNeverClass;
        else
            log()->get(LogLevel::Error) << "Shouldn't happen\n";
    }

    log()->get(LogLevel::Debug)
        << "Radius: " << m_radius << ", thresh: " << m_thresh
        << ", lambda: " << m_lambda << ", ground: " << numGround
        << ", unclass: " << numUnclass << ", never: " << numNeverClass
        << ", total: " << view.size() << std::endl;
}

void MinSampleFilter::densifyGround(PointView& view, PointViewPtr gView,
                                    const KD2Index& index)
{
    KD2Index& gIndex = gView->build2dIndex();
    for (PointRef p : view)
    {
        if (p.getFieldAs<uint8_t>(Id::Classification) !=
            ClassLabel::CreatedNeverClassified)
            continue;

        auto CalcRbfValue = [](const VectorXd& xi, const VectorXd& xj) {
            double r = (xj - xi).norm();
            double value = r * r * std::log(r);
            // std::cerr << "r: " << r << ", rbf: " << value << std::endl;
            return std::isnan(value) ? 0.0 : value;
        };
        PointIdList ids = gIndex.neighbors(p, m_count);
        MatrixXd X = MatrixXd::Zero(2, m_count);
        VectorXd y = VectorXd::Zero(m_count);
        VectorXd x = VectorXd::Zero(2);
        for (int i = 0; i < m_count; ++i)
        {
            X(0, i) = gView->getFieldAs<double>(Id::X, ids[i]);
            X(1, i) = gView->getFieldAs<double>(Id::Y, ids[i]);
            y(i) = gView->getFieldAs<double>(Id::Z, ids[i]);
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

        // log()->get(LogLevel::Debug) << "X: " << X << std::endl << std::endl;
        // log()->get(LogLevel::Debug) << "Phi: " << Phi << std::endl <<
        // std::endl; log()->get(LogLevel::Debug) << "A: " << A << std::endl <<
        // std::endl; log()->get(LogLevel::Debug) << "y: " << y << std::endl <<
        // std::endl; log()->get(LogLevel::Debug) << "b: " << b << std::endl <<
        // std::endl; log()->get(LogLevel::Debug) << "w: " << w << std::endl <<
        // std::endl;
        double val(0.0);
        for (int i = 0; i < m_count; ++i)
        {
            val += w(i) * CalcRbfValue(x, X.col(i));
        }
        double residual = p.getFieldAs<double>(Id::Z) - val;
        if (residual < m_thresh)
        {
            PointIdList ids = index.radius(p, m_radius);

            p.setField(Id::Classification, ClassLabel::Ground);
            gView->appendPoint(view, p.pointId());

            // We now proceed to mask all neighbors within m_radius of the kept
            // point.
            for (PointId const& j : ids)
            {
                if (j == p.pointId())
                    continue;
                view.setField(Id::Classification, j, ClassLabel::Unclassified);
            }
        }
    }

    point_count_t numGround = 0;
    point_count_t numUnclass = 0;
    point_count_t numNeverClass = 0;
    for (PointRef p : view)
    {
        uint8_t cls = p.getFieldAs<uint8_t>(Id::Classification);
        if (cls == ClassLabel::Ground)
            ++numGround;
        else if (cls == ClassLabel::Unclassified)
            ++numUnclass;
        else if (cls == ClassLabel::CreatedNeverClassified)
            ++numNeverClass;
        else
            log()->get(LogLevel::Error) << "Shouldn't happen\n";
    }

    log()->get(LogLevel::Debug)
        << "Radius: " << m_radius << ", thresh: " << m_thresh
        << ", lambda: " << m_lambda << ", ground: " << numGround
        << ", unclass: " << numUnclass << ", never: " << numNeverClass
        << ", total: " << view.size() << std::endl;
}

void MinSampleFilter::filter(PointView& inView)
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

    inView.calculateBounds(m_bounds);
    double xrange = m_bounds.maxx - m_bounds.minx;
    double yrange = m_bounds.maxy - m_bounds.miny;
    double zrange = m_bounds.maxz - m_bounds.minz;
    double maxrange = 2 * std::max(xrange, std::max(yrange, zrange));

    m_radius /= maxrange;
    m_thresh /= maxrange;

    PointIdList ids(inView.size());
    std::iota(ids.begin(), ids.end(), 0);
    Vector3d centroid = math::computeCentroid(inView, ids);

    for (PointRef p : inView)
    {
        double x = p.getFieldAs<double>(Id::X) - centroid.x();
        double y = p.getFieldAs<double>(Id::Y) - centroid.y();
        double z = p.getFieldAs<double>(Id::Z) - centroid.z();
        p.setField(Id::X, x / maxrange);
        p.setField(Id::Y, y / maxrange);
        p.setField(Id::Z, z / maxrange);
    }

    // Build the 2D KD-tree. Important that this comes after the sort!
    const KD2Index& index = inView.build2dIndex();

    for (PointRef p : inView)
        p.setField(Id::Classification, ClassLabel::CreatedNeverClassified);

    // All points are marked as kept (1) by default. As they are masked by
    // neighbors within the user-specified radius, their value is changed to 0.
    log()->get(LogLevel::Info) << "Finding seed points\n";
    PointViewPtr gView = maskNeighbors(inView, index);

    for (int iter = 0; iter < m_maxiters; ++iter)
    {
        m_radius *= m_radDecay;
        maskGroundNeighbors(inView, index);
        densifyGround(inView, gView, index);
        m_thresh *= m_threshDecay;
        m_lambda *= m_lambdaDecay;
    }

    for (PointRef p : inView)
    {
        double x = p.getFieldAs<double>(Id::X) * maxrange;
        double y = p.getFieldAs<double>(Id::Y) * maxrange;
        double z = p.getFieldAs<double>(Id::Z) * maxrange;
        p.setField(Id::X, x + centroid.x());
        p.setField(Id::Y, y + centroid.y());
        p.setField(Id::Z, z + centroid.z());
    }
}

} // namespace pdal
