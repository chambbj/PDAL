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

#include "RKDFilter.hpp"

#include <pdal/KDIndex.hpp>
#include <pdal/pdal_macros.hpp>

#include <Eigen/Dense>

#include <string>
#include <vector>

namespace pdal
{

static PluginInfo const s_info =
    PluginInfo("filters.rkd", "RKD Filter",
               "http://pdal.io/stages/filters.rkd.html");

CREATE_STATIC_PLUGIN(1, 0, RKDFilter, Filter, s_info)

struct peak {
    double loc;
    double area;
};

struct by_area {
    bool operator()(peak const& a, peak const& b) {
        return a.area > b.area;
    }
};

std::string RKDFilter::getName() const
{
    return s_info.name;
}

void RKDFilter::addArgs(ProgramArgs& args)
{
    args.add("bw", "Bandwidth", m_bw, 0.6);
    args.add("hres", "Horizontal resolution", m_hres, 0.05);
    args.add("vres", "Vertical resolution", m_vres, 0.05);
    args.add("radius", "Radius", m_radius, 0.15);
}

void RKDFilter::addDimensions(PointLayoutPtr layout)
{
    using namespace Dimension;
    m_rangeDensity = layout->registerOrAssignDim("Density", Type::Double);
    layout->registerDim(Id::Amplitude);
    layout->registerDim(Id::Reflectance);
    layout->registerDim(Id::ReturnNumber);
    layout->registerDim(Id::NumberOfReturns);
}

PointViewSet RKDFilter::run(PointViewPtr view)
{
    using namespace Eigen;
    using namespace Dimension;

    typedef std::vector<PointId> PointIdVec;

    // The 2D index is used to find neighbors in a Z column.
    KD2Index kd2(*view);
    kd2.build();

    // The 3D index is used to find the nearest neighbor.
    KD3Index kd3(*view);
    kd3.build();

    PointViewSet viewSet;

    // Create the output PointView.
    PointViewPtr output = view->makeNew();

    // Compute bounds to derive number of rows, cols, and samples.
    BOX3D bounds;
    view->calculateBounds(bounds);
    double extentx = bounds.maxx - bounds.minx;
    double extenty = bounds.maxy - bounds.miny;
    double extentz = bounds.maxz - bounds.minz;
    int cols = static_cast<int>(std::ceil(extentx / m_hres));
    int rows = static_cast<int>(std::ceil(extenty / m_hres));
    int n = static_cast<int>(std::ceil(extentz / m_vres));
    log()->get(LogLevel::Debug) << "# samples = " << n << std::endl;
    log()->get(LogLevel::Debug) << "# rows = " << rows << std::endl;
    log()->get(LogLevel::Debug) << "# cols = " << cols << std::endl;

    // All my vectors are out here b/c I thought there may be some savings in just allocating them once, and resizing, resetting as needed in the loops. Not sure it made that much of a difference...
    VectorXd MAPCPNeighbors = VectorXd::Zero(n);
    VectorXd density = VectorXd::Zero(n);
    // VectorXd x_vals, y_vals/*, z_vals*/;
    // VectorXd x_diff, y_diff/*, z_diff, temp*/;
    VectorXd vals = VectorXd::Zero(n-2);
    VectorXd peaks = VectorXd::Zero(n-2);
    VectorXd area = VectorXd::Zero(n-2);
    VectorXd areaFrac = VectorXd::Zero(n-2);
    VectorXd diff = VectorXd::Zero(n-1);
    VectorXd sign = VectorXd::Zero(n-1);
    VectorXd diff2 = VectorXd::Zero(n-2);

    // Initialize the samples.
    VectorXd samples(n);
    for (int i = 0; i < samples.size(); ++i)
        samples(i) = bounds.minz + i * m_vres;

    for (int r = 0; r < rows; ++r)
    {
        double y = bounds.miny + r * m_hres;
        for (int c = 0; c < cols; ++c)
        {
            double x = bounds.minx + c * m_hres;

            // Find neighbors in raw cloud at current XY cell.
            PointIdVec neighbors = kd2.radius(x, y, m_radius);

            // Record values from each of the neighbors.
            // x_vals.resize(neighbors.size());
            // y_vals.resize(neighbors.size());
            // z_vals.resize(neighbors.size());
            ArrayXf x_vals(neighbors.size());
            ArrayXf y_vals(neighbors.size());
            ArrayXf z_vals(neighbors.size());
            
            // long long n_x, n_y, n_z;
            // double M1_x, M1_y, M1_z;
            // double M2_x, M2_y, M2_z;
            // n_x = n_y = n_z = 0;
            // M1_x = M1_y = M1_z = 0.0;
            // M2_x = M2_y = M2_z = 0.0;

            for (PointId idx = 0; idx < neighbors.size(); ++idx)
            {
                x_vals(idx) = view->getFieldAs<double>(Id::X, neighbors[idx]);
                y_vals(idx) = view->getFieldAs<double>(Id::Y, neighbors[idx]);
                z_vals(idx) = view->getFieldAs<double>(Id::Z, neighbors[idx]);
                
                // double delta_x, delta_x_n, delta_y, delta_y_n, delta_z, delta_z_n;
                // double term1_x, term1_y, term1_z;
                // long long n1_x = n_x;
                // long long n1_y = n_y;
                // long long n1_z = n_z;
                // 
                // n_x++;
                // n_y++;
                // n_z++;
                // 
                // delta_x = x_vals(idx) - M1_x;
                // delta_x_n = delta_x / n_x;
                // term1_x = delta_x * delta_x_n * n1_x;
                // M1_x += delta_x_n;
                // M2_x += term1_x;
                // 
                // delta_y = y_vals(idx) - M1_y;
                // delta_y_n = delta_y / n_y;
                // term1_y = delta_y * delta_y_n * n1_y;
                // M1_y += delta_y_n;
                // M2_y += term1_y;
                // 
                // delta_z = z_vals(idx) - M1_z;
                // delta_z_n = delta_z / n_z;
                // term1_z = delta_z * delta_z_n * n1_z;
                // M1_z += delta_z_n;
                // M2_z += term1_z;
            }

            // x_diff.resize(neighbors.size());
            // y_diff.resize(neighbors.size());
            // z_diff.resize(neighbors.size());
            // temp.resize(neighbors.size());
            
            // std::cerr << std::fixed << std::setprecision(8);
            // std::cerr << "X: " << x_vals.transpose() << std::endl;
            // std::cerr << "Y: " << y_vals.transpose() << std::endl;
            // std::cerr << "Z: " << z_vals.transpose() << std::endl;
            
            // double h_x = std::sqrt(M2_x / (neighbors.size()-1.0)) * 2.34 * std::pow(neighbors.size(), -1.0/5.0);
            // double h_y = std::sqrt(M2_y / (neighbors.size()-1.0)) * 2.34 * std::pow(neighbors.size(), -1.0/5.0);
            // double h_z = std::sqrt(M2_z / (neighbors.size()-1.0)) * 2.34 * std::pow(neighbors.size(), -1.0/5.0);
            double h_x, h_y, h_z;
            h_x = h_y = 0.15 * 2.34 * std::pow(neighbors.size(), -1.0/5.0);
            h_z = 0.3 * 2.34 * std::pow(neighbors.size(), -1.0/5.0);
            // std::cerr << h_x << "\t" << h_y << "\t" << h_z << std::endl;

            // Sample density for the current column.
            // double invbw = 1 / m_bw;
            double factor = 0.75 / (neighbors.size() * h_x * h_y * h_z);
            
            ArrayXf x_diff = x_vals - x;
            x_diff /= h_x;
            ArrayXf xx = 1 - x_diff.cwiseProduct(x_diff);
            xx = (x_diff.array().abs() > 1).select(0, xx);
            
            ArrayXf y_diff = y_vals - y;
            y_diff /= h_y;
            ArrayXf yy = 1 - y_diff.cwiseProduct(y_diff);
            yy = (y_diff.array().abs() > 1).select(0, yy);
            
            ArrayXf xyprod = xx * yy;
            
            for (size_t i = 0; i < samples.size(); ++i)
            {                
                ArrayXf z_diff = z_vals - samples(i);
                z_diff /= h_z;
                ArrayXf zz = 1 - z_diff * z_diff;
                zz = (z_diff.abs() > 1).select(0, zz);
                
                // if ((z_diff.array() > 1).count())
                    // std::cerr << zz.transpose() << std::endl;
                
                // std::cerr << (x_diff.array() > 1.0).count() << "\t"
                //           << (y_diff.array() > 1.0).count() << "\t"
                //           << (z_diff.array() > 1.0).count() << "\t"
                //           << std::endl;
                
                // temp = xx.cwiseProduct(yy);
                ArrayXf temp = xyprod * zz;
                // std::cerr << xx.size() << "\t" << yy.size() << "\t" << zz.size() << "\t" << temp.size() << std::endl;
                density(i) = factor * temp.sum();
            }
            // how critical is it to normalize this in some way? it does affect the peak area, the overall area, and therefor the intensity and reflectance
            density /= density.sum();
            // std::cerr << density.sum() << "\t" << density.norm() << "\t" << density.maxCoeff() << std::endl;
            // std::cerr << density.sum() << std::endl;
            // density.normalize();
            // std::cerr << density.sum() << std::endl;

            // std::cerr << "peaks\n";
            // std::cerr << samples.transpose() << std::endl;
            // std::cerr << "density\n";
            // std::cerr << density.transpose() << std::endl;

            auto diffEq = [](VectorXd vec)
            {
                return vec.tail(vec.size()-1)-vec.head(vec.size()-1);
            };

            // MATLAB diff command - approximate derivative
            diff = diffEq(density);

            // MATLAB sign command - sigmoid function
            for (int i = 0; i < samples.size()-1; ++i)
            {
                if (diff(i) < 0)
                    sign(i) = -1;
                else if (diff(i) > 0)
                    sign(i) = 1;
                else
                    sign(i) = 0;
            }

            // MATLAB diff command again - approxiate derivative
            diff2 = diffEq(sign);

            // std::cerr << "diff2\n";
            // std::cerr << diff2.transpose() << std::endl;

            vals.resize(n-2);
            peaks.resize(n-2);
            area.resize(n-2);
            areaFrac.resize(n-2);
            
            double invdensitysum = 1 / density.sum();
            
            // Peaks occur at diff2 == -2
            int nPeaks = 0;
            int nrad = 3;
            double rad = (2*nrad+1)*m_hres/2;
            double totPeakArea = 0.0;
            std::vector<peak> pvec;
            for (int i = 0; i < samples.size()-2; ++i)
            {
                if (diff2(i) == -2)
                {
                    // int nei = kd3.radius(x, y, samples(i), rad).size();
                    // if (nei < 3)
                    //     continue;
                        
                    double peakArea = density(i);
                    for (int j = i+1; j < samples.size()-2; ++j)
                    {
                        if (diff2(j) > 0)
                            break;
                        peakArea += density(j);
                    }
                    for (int j = i-1; j >= 0; --j)
                    {
                        if (diff2(j) > 0)
                            break;
                        peakArea += density(j);
                    }
                    // if (peakArea * invdensitysum < 0.1)
                    //     continue;

                    // vals(nPeaks) = density(i);
                    // vals(nPeaks) = nei;  // experiment, write number of neighbors out to density channel
                    // peaks(nPeaks) = samples(i);
                    // area(nPeaks) = peakArea;
                    // areaFrac(nPeaks) = peakArea * invdensitysum;
                    
                    // totPeakArea += peakArea;
                    pvec.push_back(peak{samples(i), peakArea});

                    nPeaks++;
                }
            }
            
            std::sort(pvec.begin(), pvec.end(), by_area());

            if (nPeaks == 0)
                continue;

            // vals.conservativeResize(nPeaks);
            // peaks.conservativeResize(nPeaks);
            // area.conservativeResize(nPeaks);
            // areaFrac.conservativeResize(nPeaks);
            
            // std::cerr << vals.transpose() << std::endl;
            // std::cerr << peaks.transpose() << std::endl;
            // std::cerr << "area\n";
            // std::cerr << area.transpose() << std::endl;
            // std::cerr << areaFrac.transpose() << std::endl;

            // vals /= vals.sum();
            // double maxval = vals.maxCoeff();
            // double peakAreaAdded = 0.0;

            // For each peak of sufficient size/strength, find the nearest
            // neighbor in the raw data and append to the output view.
            // for (int i = 0; i < nPeaks; ++i)
            // int np = 0;
            for (auto const& p : pvec)
            {
                if (p.area < 0.1)
                    break;
                // if (++np > 2)
                //     continue;
                // if (peakAreaAdded > (0.8 * totPeakArea))
                //     continue;
                    
                // peakAreaAdded += p.area;
                // if (vals(i) < maxval)
                //     continue;
                    
                PointIdVec idx = kd3.neighbors(x, y, p.loc, 1);
                // view->setField(m_rangeDensity, idx[0], vals(i));
                // view->setField(Id::NumberOfReturns, idx[0], nPeaks);
                // view->setField(Id::ReturnNumber, idx[0], nPeaks-i);
                view->setField(Id::Amplitude, idx[0], p.area);
                // view->setField(Id::Reflectance, idx[0], areaFrac(i));
                output->appendPoint(*view, idx[0]);
                /*
                PointIdVec idx = kd3.radius(x, y, peaks(i), m_hres);
                for (auto const& id : idx)
                {
                    view->setField(m_rangeDensity, id, vals(i));
                    view->setField(Id::NumberOfReturns, id, nPeaks);
                    view->setField(Id::ReturnNumber, id, nPeaks-i);
                    view->setField(Id::Amplitude, id, area(i));
                    view->setField(Id::Reflectance, id, areaFrac(i));
                    output->appendPoint(*view, id);
                }
                */
            }
        }
    }

    viewSet.erase(view);
    viewSet.insert(output);
    return viewSet;
}

} // namespace pdal
