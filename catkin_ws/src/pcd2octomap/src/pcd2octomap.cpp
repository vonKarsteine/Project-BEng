// Convert a .pcd point cloud (e.g. FAST-LIO's PCD/scans.pcd) into an
// octomap .bt occupancy tree.
//
//   pcd2octomap input.pcd output.bt [resolution=0.05]
//
// Visualize the result with `octovis output.bt`.

#include <iostream>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <octomap/octomap.h>

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0] << " input.pcd output.bt [resolution=0.05]" << std::endl;
        return 1;
    }
    const std::string input_file = argv[1];
    const std::string output_file = argv[2];
    const double resolution = (argc > 3) ? std::stod(argv[3]) : 0.05;

    pcl::PointCloud<pcl::PointXYZ> cloud;
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_file, cloud) < 0)
    {
        std::cerr << "failed to load " << input_file << std::endl;
        return 1;
    }
    std::cout << "loaded " << cloud.points.size() << " points from " << input_file << std::endl;

    octomap::OcTree tree(resolution);
    for (const auto &p : cloud.points)
    {
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
            tree.updateNode(octomap::point3d(p.x, p.y, p.z), true);
    }
    tree.updateInnerOccupancy();

    if (!tree.writeBinary(output_file))
    {
        std::cerr << "failed to write " << output_file << std::endl;
        return 1;
    }
    std::cout << "wrote " << output_file << " (resolution " << resolution << " m)" << std::endl;
    return 0;
}
