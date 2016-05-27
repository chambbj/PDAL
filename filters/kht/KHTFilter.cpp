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

#include <cmath>

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

std::vector<PointId> KHTFilter::refineFit(PointViewPtr view, std::vector<PointId> ids, Eigen::Vector3d centroid, Eigen::Vector3d normal, double threshold)
{
    using namespace Eigen;

    // Create a plane with given normal (equal to eigenvector corresponding to
    // smallest eigenvalue) and passing through the centroid.
    auto plane = Hyperplane<double, 3>(normal, 0);

    // Iterate over node indices and only keep those that are within a given
    // tolerance of the plane surface.
    std::vector<PointId> new_ids;
    for (auto const& j : ids)
    {
        Vector3d pt(view->getFieldAs<double>(Dimension::Id::X, j),
                    view->getFieldAs<double>(Dimension::Id::Y, j),
                    view->getFieldAs<double>(Dimension::Id::Z, j));
        pt = pt - centroid;
        if (plane.absDistance(pt) < threshold)
            new_ids.push_back(j);
    }
    return new_ids;
}

Eigen::Matrix3d KHTFilter::computeJacobian(Eigen::Vector3d centroid, Eigen::Vector3d normal)
{
    using namespace Eigen;

    double rho = centroid.dot(normal);
    double rho2 = rho*rho;
    Vector3d p = rho * normal;
    double angle = std::acos(rho/(centroid.norm()*normal.norm()));
    std::cerr << "Angle is " << angle*180/3.14159 << std::endl;
    if (angle*180/3.14159 > 90)
    {
        normal *= -1;
        std::cerr << "Flipping normal to " << normal.transpose() << std::endl;
        rho = centroid.dot(normal);
        rho2 = rho*rho;
        p = rho * normal;
    }
    auto w = p.x()*p.x()+p.y()*p.y();
    auto rootw = std::sqrt(w);
    Matrix3d J;
    J.row(0) << normal.x(), normal.y(), normal.z();
    J.row(1) << p.x()*p.z()/(rootw*rho2), p.y()*p.z()/(rootw*rho2), -rootw/rho2;
    J.row(2) << -p.y()/w, p.x()/w, 0;

    return J;
}

void KHTFilter::cluster(PointViewPtr view, std::vector<PointId> ids, int level)
{
    using namespace Eigen;

    bool coplanar(false);
    point_count_t np(ids.size());
    SelfAdjointEigenSolver<Matrix3d> solver;
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
    
    if (level == 0)
    {
        m_totalArea = xEdge * yEdge * zEdge;
        m_totalPoints = view->size();
        std::cerr << m_totalPoints << ", " << m_totalArea << std::endl;
    }

    // Don't even begin looking for clusters of coplanar points until we reach a
    // given level in the octree.
    if (level > 3)
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

            // Refine the planar fit and use the updated ids going forward.
            auto new_ids = refineFit(view, ids, centroid, normal, threshold);
            ids.swap(new_ids);

            // Re-compute centroid and covariance  of the node with updated
            // indices.
            centroid = computeCentroid(*view, ids);
            B = computeCovariance(*view, centroid, ids);

            // Perform the eigen decomposition with updated indices.
            solver.compute(B);
            if (solver.info() != Success)
                throw pdal_error("Cannot perform eigen decomposition.");
            normal = solver.eigenvectors().col(0);

            // Construct the Jacobian.
            auto J = computeJacobian(centroid, normal);
            
            std::cerr << "Cov: " << B << std::endl;
            std::cerr << "Jac: " << J << std::endl;
            
            std::cerr << "sanity: " << J * B << std::endl;

            // Transform covariance to polar coordinates using Jacobian.
            Matrix3d Sigma = J * B * J.transpose();
            Sigma(0,0) += 0.001; // to avoid singular cases
            std::cerr << "Sigma: " << Sigma << std::endl;

            // Perform the eigen decomposition with the covariance in polar
            // coordinates.
            solver.compute(Sigma);
            if (solver.info() != Success)
                throw pdal_error("Cannot perform eigen decomposition.");
            auto stdev = std::sqrt(solver.eigenvalues()[0]);
            auto gmin = 2 * stdev * solver.eigenvectors().col(0);
          
            Matrix3d SigmaInv;
            double SigmaDet;
            bool invertible;
            Sigma.computeInverseAndDetWithCheck(SigmaInv, SigmaDet, invertible);
            
            if (!invertible)
            {
                std::cerr << "sigma is not invertible\n";
                return;
            }
              
            std::cerr << "Determinant: " << SigmaDet << std::endl;
            std::cerr << "Inverse: " << SigmaInv << std::endl;
            
            auto factor = 1 / (15.7496 * std::sqrt(SigmaDet));
            std::cerr << "Factor: " << factor << std::endl;
            
            auto weight = 0.75 * (xEdge * yEdge * zEdge / m_totalArea) + 0.25 * (ids.size() / m_totalPoints);
            std::cerr << "Weight: " << weight << std::endl;
            
            // compute centroid of polar coordinates
            double rhoSum = 0.0;
            double thetaSum = 0.0;
            double phiSum = 0.0;
            for (auto const& j : ids)
            {
                Vector3d pt(view->getFieldAs<double>(Dimension::Id::X, j),
                            view->getFieldAs<double>(Dimension::Id::Y, j),
                            view->getFieldAs<double>(Dimension::Id::Z, j));
                pt = pt - centroid;
                auto rho = std::sqrt(pt.x()*pt.x()+pt.y()*pt.y()+pt.z()*pt.z());
                rhoSum += rho;
                thetaSum += std::atan(pt.y()/pt.x());
                phiSum += std::acos(pt.z()/rho);
            }
            Vector3d polarCentroid(rhoSum/ids.size(), phiSum/ids.size(), thetaSum/ids.size());
            
            for (auto const& j : ids)
            {
                Vector3d pt(view->getFieldAs<double>(Dimension::Id::X, j),
                            view->getFieldAs<double>(Dimension::Id::Y, j),
                            view->getFieldAs<double>(Dimension::Id::Z, j));
                pt = pt - centroid;
                auto rho = std::sqrt(pt.x()*pt.x()+pt.y()*pt.y()+pt.z()*pt.z());
                auto theta = std::atan(pt.y()/pt.x());
                auto phi = std::acos(pt.z()/rho);
                Vector3d q(rho, phi, theta);
                q = q - polarCentroid;
                auto temp = -0.5 * q.transpose() * SigmaInv * q;
                auto temp2 = factor * std::exp(temp);
                auto temp3 = weight * temp2;
                // std::cerr << q.transpose() << std::endl;
                // std::cerr << std::exp(temp) << std::endl;
                // printf("%f.8\n", temp3);
                printf("Bin %.2f %.2f %.2f gets a vote of %.8f\n", theta*180/3.14, phi*180/3.14, rho, temp3);
            }
            
            // std::cerr << gmin << std::endl;

            log()->get(LogLevel::Debug2)
                << "Level: " << level << "\t"
                << "Points before: " << np << "\t"
                << "Points after: " << ids.size() << "\t"
                << "Stdev: " << stdev << std::endl;

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
