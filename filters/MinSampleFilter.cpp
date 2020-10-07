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
}

void MinSampleFilter::addDimensions(PointLayoutPtr layout)
{
    layout->registerDim(Id::HeightAboveGround);
}

PointViewPtr MinSampleFilter::maskNeighbors(PointView& view,
                                            const KD2Index& index,
                                            std::vector<int>& keep)
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
        if (keep[p.pointId()] == 0)
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
            keep[j] = 0;
            view.setField(Id::Classification, j, ClassLabel::Unclassified);
        }
    }

    //log()->get(LogLevel::Debug) << "At radius of " << m_radius << " kept "
    //                            << std::accumulate(keep.begin(), keep.end(), 0)
    //                            << " points for initial ground surface\n";
    log()->get(LogLevel::Debug) << "Radius: " << m_radius
	                        << ", thresh: " << m_thresh
				<< ", kept: " << std::accumulate(keep.begin(), keep.end(), 0)
				<< ", ground: " << gView->size()
				<< ", total: " << view.size() << std::endl;

    return gView;
}

void MinSampleFilter::maskGroundNeighbors(PointView& view,
                                          const KD2Index& index,
                                          std::vector<int>& keep)
{
    point_count_t num = 0;
    for (PointRef p : view)
    {
        if (p.getFieldAs<double>(Id::Classification) == ClassLabel::Ground)
        {
            ++num;
            PointIdList ids = index.radius(p, m_radius);

            // We now proceed to mask all neighbors within m_radius of the kept
            // point.
            for (PointId const& j : ids)
            {
                if (j == p.pointId())
                    continue;
                keep[j] = 0;
            }
        }
    }

    //log()->get(LogLevel::Debug) << "At radius of " << m_radius << " there are "
    //                            << std::accumulate(keep.begin(), keep.end(), 0)
    //                            << " candidate ground points\n";
    //log()->get(LogLevel::Debug)
    //    << "This indicates that of the " << view.size()
    //    << " points, the number masked by " << num << " ground returns is "
    //    << view.size() - std::accumulate(keep.begin(), keep.end(), 0)
    //    << std::endl;
}

void MinSampleFilter::densifyGround(PointView& view, PointViewPtr gView,
                                    const KD2Index& index,
                                    std::vector<int>& keep)
{
    KD2Index& gIndex = gView->build2dIndex();
    point_count_t numAdded = 0;
    for (PointRef p : view)
    {
        if (keep[p.pointId()] == 0)
            continue;

        auto CalcRbfValue = [](const VectorXd& xi, const VectorXd& xj) {
            double r = (xj - xi).norm();
            double value = r * r * std::log(r);
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
        VectorXd w = Eigen::PartialPivLU<MatrixXd>(Phi).solve(y);
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
                keep[j] = 0;
            }
        }
    }
    //log()->get(LogLevel::Debug)
    //    << "Ground surface has " << std::accumulate(keep.begin(), keep.end(), 0)
    //    << " points after densification with threshold " << m_thresh
    //    << std::endl;
    log()->get(LogLevel::Debug) << "Radius: " << m_radius
	                        << ", thresh: " << m_thresh
				<< ", kept: " << std::accumulate(keep.begin(), keep.end(), 0)
				<< ", ground: " << gView->size()
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

    // Build the 2D KD-tree. Important that this comes after the sort!
    const KD2Index& index = inView.build2dIndex();

    // All points are marked as kept (1) by default. As they are masked by
    // neighbors within the user-specified radius, their value is changed to 0.
    log()->get(LogLevel::Info) << "Finding seed points\n";
    std::vector<int> keep(inView.size(), 1);
    PointViewPtr gView = maskNeighbors(inView, index, keep);

    for (int iter = 0; iter < m_maxiters; ++iter)
    {
        keep.assign(inView.size(), 1);
        m_radius *= 0.5;
        m_thresh *= 0.9;
        maskGroundNeighbors(inView, index, keep);
        densifyGround(inView, gView, index, keep);
    }
}

} // namespace pdal
