/******************************************************************************
* Copyright (c) 2016, Bradley J Chambers (brad.chambers@gmail.com)
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

#include "KHTFilter.hpp"

#include <pdal/Eigen.hpp>
#include <pdal/pdal_macros.hpp>

#include <Eigen/Dense>

namespace pdal
{

static PluginInfo const s_info =
    PluginInfo("filters.kht", "3D-KHT filter",
               "http://pdal.io/stages/filters.kht.html");

CREATE_STATIC_PLUGIN(1, 0, KHTFilter, Filter, s_info)

std::string KHTFilter::getName() const
{
    return s_info.name;
}

Options KHTFilter::getDefaultOptions()
{
    Options options;
    // options.add("max_window_size", 33, "Maximum window size");
    return options;
}

void KHTFilter::processOptions(const Options& options)
{
    // m_maxWindowSize = options.getValueOrDefault<double>("max_window_size", 33);
}

void KHTFilter::addDimensions(PointLayoutPtr layout)
{
    layout->registerDim(Dimension::Id::Classification);
}

void KHTFilter::cluster(PointViewPtr view, std::vector<PointId> ids, int level)
{
    using namespace Eigen;

    bool coplanar(false);
    point_count_t np(ids.size());
    SelfAdjointEigenSolver<Matrix3f> solver;
    std::vector<PointId> original_ids(ids);

    // If there are too few points in the current node, then bail. We need a
    // minimum number of points to determine whether or not the points are
    // coplanar and if this cluster can be used for voting.
    if (np < 30)
        return;

    // Compute the bounds of the current node. (Scoped so that we will let go of
    // the PointViewPtr.)
    BOX3D bounds;
    {
        // Create a temporary PointView for the current node.
        PointViewPtr node = view->makeNew();
        for (auto const& i : ids)
            node->appendPoint(*view, i);

        node->calculateBounds(bounds);
    }

    // Compute some stats on the current node.
    double xEdge = (bounds.maxx - bounds.minx);
    double xSplit = xEdge/2 + bounds.minx;
    double yEdge = (bounds.maxy - bounds.miny);
    double ySplit = yEdge/2 + bounds.miny;
    double zEdge = (bounds.maxz - bounds.minz);
    double zSplit = zEdge/2 + bounds.minz;
    double maxEdge = std::max(xEdge, std::max(yEdge, zEdge));
    double threshold = maxEdge / 10;

    // Don't even begin looking for clusters of coplanar points until we reach a
    // given level in the octree.
    if (level > 4)
    {
        // Compute centroid of the node (as given by ids).
        auto centroid = computeCentroid(*view, ids);
        
        // Compute covariance of the node (as given by ids).
        auto B = computeCovariance(*view, centroid, ids);

        // Perform the eigen decomposition.
        solver.compute(B);
        if (solver.info() != Success)
            throw pdal_error("Cannot perform eigen decomposition.");
        auto eigVal = solver.eigenvalues();
        auto eigVec = solver.eigenvectors();
        auto normal = eigVec.col(0);

        // Test plane thickness and isotropy.
        if ((eigVal[1] > 25 * eigVal[0]) && (6 * eigVal[1] > eigVal[2]))
        {
            coplanar = true;
            
            auto refineFit = [&view, &ids, &centroid, &normal, &threshold]()
            {
                // Create a plane with given normal (equal to eigenvector 
                // corresponding to smallest eigenvalue) and passing through the
                // centroid.
                auto plane = Hyperplane<float, 3>(normal, 0);
                
                // Iterate over node indices and only keep those that are within
                // a given tolerance of the plane surface.
                std::vector<PointId> new_ids;
                for (auto const& j : ids)
                {
                    Vector3f pt(view->getFieldAs<double>(Dimension::Id::X, j),
                                view->getFieldAs<double>(Dimension::Id::Y, j),
                                view->getFieldAs<double>(Dimension::Id::Z, j));
                    pt = pt - centroid;
                    if (plane.absDistance(pt) < threshold)
                        new_ids.push_back(j);
                }
                return new_ids;
            };
            
            auto new_ids = refineFit();
            
            // Use the new ids from this point forward.
            ids.swap(new_ids);
                
            // Re-compute centroid of the node with updated indices.
            centroid = computeCentroid(*view, ids);
            
            // Re-compute covariance of the node with updated indices.
            // TODO(chambbj): we very often compute centroid separately, even
            // though it is already computed here - maybe we should require
            // centroid as a parameter - while the user will be forced to
            // compute centroid, she will also have it available for reuse
            B = computeCovariance(*view, centroid, ids);
            
            // Perform the eigen decomposition once again with updated 
            // indices.
            solver.compute(B);
            if (solver.info() != Success)
                throw pdal_error("Cannot perform eigen decomposition.");
            eigVec = solver.eigenvectors();
            normal = eigVec.col(0);

            // TODO(chambbj): consider how general purpose this could be, while
            // this works...what _should_ it look like
            auto computeJacobian = [&centroid, &normal]()
            {    
                auto rho = centroid.dot(normal);
                auto rho2 = rho*rho;
                auto p = rho * normal;
                auto w = p.x()*p.x()+p.y()*p.y();
                auto rootw = std::sqrt(w);
                Matrix3f J;
                J.row(0) << normal.x(), normal.y(), normal.z();
                J.row(1) << p.x()*p.z()/(rootw*rho2),
                      p.y()*p.z()/(rootw*rho2),
                      -rootw/rho2;
                J.row(2) << -p.y()/w, p.x()/w, 0;
                
                return J;
            };

            // Construct the Jacobian.
            auto J = computeJacobian();
            
            // Transform covariance to polar coordinates using Jacobian.
            auto Sigma = J * B * J.transpose();

            // Perform the eigen decomposition once again with the covariance in
            // polar coordinates.
            solver.compute(Sigma);
            if (solver.info() != Success)
                throw pdal_error("Cannot perform eigen decomposition.");
            eigVal = solver.eigenvalues();
            eigVec = solver.eigenvectors();
            auto stdev = std::sqrt(eigVal[0]);

            std::cerr << "found candidate coplanar at level " << level
                      << " with " << np << " points before and "
                      << ids.size() << " points after refinement\n";
            std::cerr << Sigma << std::endl;
            std::cerr << "eigenvalues:\n";
            std::cerr << eigVal << std::endl;
            std::cerr << "eigenvectors:\n";
            std::cerr << eigVec << std::endl;
            std::cerr << "threshold will be " << stdev << std::endl;

            return;
        }
    }

    // Create eight vectors of points indices for the eight children of the
    // current node.
    std::vector<PointId> upNWView, upNEView, upSEView, upSWView, downNWView,
        downNEView, downSEView, downSWView;
        
    // Populate the children, using original_ids as we may have removed points
    // from the ids vector during refinement.
    for (auto const& i : original_ids)
    {
        double x = view->getFieldAs<double>(Dimension::Id::X, i);
        double y = view->getFieldAs<double>(Dimension::Id::Y, i);
        double z = view->getFieldAs<double>(Dimension::Id::Z, i);

        if (x < xSplit && y >= ySplit && z >= zSplit)
            upNWView.push_back(i);
        if (x >= xSplit && y >= ySplit && z >= zSplit)
            upNEView.push_back(i);
        if (x >= xSplit && y < ySplit && z >= zSplit)
            upSEView.push_back(i);
        if (x < xSplit && y < ySplit && z >= zSplit)
            upSWView.push_back(i);
        if (x < xSplit && y >= ySplit && z < zSplit)
            downNWView.push_back(i);
        if (x >= xSplit && y >= ySplit && z < zSplit)
            downNEView.push_back(i);
        if (x >= xSplit && y < ySplit && z < zSplit)
            downSEView.push_back(i);
        if (x < xSplit && y < ySplit && z < zSplit)
            downSWView.push_back(i);
    }

    std::vector<std::vector<PointId>> children
    {
        upNWView, upNEView, upSEView, upSWView, downNWView, downNEView,
        downSEView, downSWView
    };

    // Loop over children and recursively cluster at the next level in the
    // octree.
    for (auto const& child : children)
        cluster(view, child, level+1);

    // if cluster is coplanar, cast votes, etc.
}

PointViewSet KHTFilter::run(PointViewPtr view)
{
    PointViewSet viewSet;

    bool logOutput = log()->getLevel() > LogLevel::Debug1;
    if (logOutput)
        log()->floatPrecision(8);
    log()->get(LogLevel::Debug2) << "Process KHTFilter...\n";

    std::vector<PointId> ids;
    for (PointId i = 0; i < view->size(); ++i)
        ids.push_back(i);

    cluster(view, ids, 0);

    viewSet.insert(view);

    return viewSet;
}

} // namespace pdal
