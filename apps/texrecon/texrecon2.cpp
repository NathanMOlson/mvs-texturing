/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <tbb/task_arena.h>
#include <omp.h>

#include <util/timer.h>
#include <util/system.h>
#include <util/file_system.h>
#include <mve/mesh_io_ply.h>
#include <opencv2/imgcodecs.hpp>

#include "tex/util.h"
#include "tex/timer.h"
#include "tex/debug.h"
#include "tex/texturing.h"
#include "tex/progress_counter.h"

#include "arguments.h"

cv::Mat view_selection(tex::DataCosts const &data_costs,
                    const QuadMesh &mesh,
                    const std::vector<float> &cost_table,
                    tex::Settings const &);

int main(int argc, char **argv)
{
    util::system::print_build_timestamp(argv[0]);
    util::system::register_segfault_handler();

    Timer timer;
    util::WallTimer wtimer;

    Arguments conf;
    try
    {
        conf = parse_args(argc, argv);
    }
    catch (std::invalid_argument &ia)
    {
        std::cerr << ia.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string const out_dir = util::fs::dirname(conf.out_prefix);

    if (!util::fs::dir_exists(out_dir.c_str()))
    {
        std::cerr << "Destination directory does not exist!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string const tmp_dir = util::fs::join_path(out_dir, "tmp");
    if (!util::fs::dir_exists(tmp_dir.c_str()))
    {
        util::fs::mkdir(tmp_dir.c_str());
    }
    else
    {
        std::cout << "Careful! Temporary directory \"tmp\" exists within the destination directory." << std::endl;
    }

    // Set the number of threads to use.
    tbb::task_arena arena(conf.num_threads > 0 ? conf.num_threads : tbb::this_task_arena::max_concurrency());

    if (conf.num_threads > 0)
    {
        omp_set_dynamic(0);
        omp_set_num_threads(conf.num_threads);
    }

    std::cout << "Load and prepare mesh: " << std::endl;
    QuadMesh mesh(conf.in_mesh);

    std::cout << "Generating texture views: " << std::endl;
    std::vector<ImageView> image_views = generate_image_views(conf.in_scene, tmp_dir);

    write_string_to_file(conf.out_prefix + ".conf", conf.to_string());
    timer.measure("Loading");

    std::size_t const num_faces = mesh.NumFaces();

    if (conf.labeling_file.empty())
    {
        std::cout << "View selection:" << std::endl;
        util::WallTimer rwtimer;

        tex::DataCosts data_costs(num_faces, image_views.size());
        if (conf.data_cost_file.empty())
        {
            calculate_data_costs(&mesh, image_views, conf.settings, &data_costs);

            if (conf.write_intermediate_results)
            {
                std::cout << "\tWriting data cost file... " << std::flush;
                tex::DataCosts::save_to_file(data_costs, conf.out_prefix + "_data_costs.spt");
                std::cout << "done." << std::endl;
            }
        }
        else
        {
            std::cout << "\tLoading data cost file... " << std::flush;
            try
            {
                tex::DataCosts::load_from_file(conf.data_cost_file, &data_costs);
            }
            catch (util::FileException e)
            {
                std::cout << "failed!" << std::endl;
                std::cerr << e.what() << std::endl;
                std::exit(EXIT_FAILURE);
            }
            std::cout << "done." << std::endl;
        }
        timer.measure("Calculating data costs");

        size_t n_views = image_views.size();
        std::vector<float> pairwise_cost(n_views * n_views, 0);
        for (size_t i = 0; i < n_views; i++)
        {
            for (size_t j = i + 1; j < n_views; j++)
            {
                pairwise_cost[i * n_views + j] = (image_views[i].get_pos() - image_views[j].get_pos()).norm() * 1.0e-3;
                pairwise_cost[j * n_views + i] = pairwise_cost[i * n_views + j];
            }
        }

        cv::Mat labels;
        try
        {
            labels = view_selection(data_costs, mesh, pairwise_cost, conf.settings);
        }
        catch (std::runtime_error &e)
        {
            std::cerr << "\tOptimization failed: " << e.what() << std::endl;
            std::exit(EXIT_FAILURE);
        }
        timer.measure("Running MRF optimization");
        std::cout << "\tTook: " << rwtimer.get_elapsed_sec() << "s" << std::endl;

        /* Write labeling to file. */
        if (conf.write_intermediate_results)
        {
            cv::imwrite(conf.out_prefix + "_labeling.png", labels);
        }
    }
    else
    {
        // std::cout << "Loading labeling from file... " << std::flush;

        // /* Load labeling from file. */
        // std::vector<std::size_t> labeling = vector_from_file<std::size_t>(conf.labeling_file);
        // if (labeling.size() != graph.num_nodes()) {
        //     std::cerr << "Wrong labeling file for this mesh/scene combination... aborting!" << std::endl;
        //     std::exit(EXIT_FAILURE);
        // }

        // /* Transfer labeling to graph. */
        // for (std::size_t i = 0; i < labeling.size(); ++i) {
        //     const std::size_t label = labeling[i];
        //     if (label > image_views.size()){
        //         std::cerr << "Wrong labeling file for this mesh/scene combination... aborting!" << std::endl;
        //         std::exit(EXIT_FAILURE);
        //     }
        //     graph.set_label(i, label);
        // }

        // std::cout << "done." << std::endl;
    }

    //     tex::TextureAtlases texture_atlases;
    //     {
    //         /* Create texture patches and adjust them. */
    //         tex::TexturePatches texture_patches;
    //         tex::VertexProjectionInfos vertex_projection_infos;
    //         std::cout << "Generating texture patches:" << std::endl;
    //         tex::generate_texture_patches(graph, mesh, mesh_info, &image_views,
    //             conf.settings, &vertex_projection_infos, &texture_patches);

    //         if (conf.settings.global_seam_leveling) {
    //             std::cout << "Running global seam leveling:" << std::endl;
    //             tex::global_seam_leveling(graph, mesh, mesh_info, vertex_projection_infos, &texture_patches);
    //             timer.measure("Running global seam leveling");
    //         } else {
    //             ProgressCounter texture_patch_counter("Calculating validity masks for texture patches", texture_patches.size());
    //             #pragma omp parallel for schedule(dynamic)
    // #if !defined(_MSC_VER)
    //             for (std::size_t i = 0; i < texture_patches.size(); ++i) {
    // #else
    //             for (std::int64_t i = 0; i < texture_patches.size(); ++i) {
    // #endif
    //                 texture_patch_counter.progress<SIMPLE>();
    //                 TexturePatch::Ptr texture_patch = texture_patches[i];
    //                 std::vector<math::Vec3f> patch_adjust_values(texture_patch->get_faces().size() * 3, math::Vec3f(0.0f));
    //                 texture_patch->adjust_colors(patch_adjust_values);
    //                 texture_patch_counter.inc();
    //             }
    //             timer.measure("Calculating texture patch validity masks");
    //         }

    //         if (conf.settings.local_seam_leveling) {
    //             std::cout << "Running local seam leveling:" << std::endl;
    //             tex::local_seam_leveling(graph, mesh, vertex_projection_infos, &texture_patches);
    //         }
    //         timer.measure("Running local seam leveling");

    //         /* Generate texture atlases. */
    //         std::cout << "Generating texture atlases:" << std::endl;

    //         bool grayscale = image_views.front().is_grayscale();
    //         tex::generate_texture_atlases(&texture_patches, conf.settings, &texture_atlases, type, grayscale);
    //     }

    //     /* Create and write out obj model. */
    //     {
    //         std::cout << "Building objmodel:" << std::endl;
    //         tex::Model model;
    //         tex::build_model(mesh, texture_atlases, &model);
    //         timer.measure("Building OBJ model");

    //         std::cout << "\tSaving model... " << std::flush;
    //         tex::Model::save(model, conf.out_prefix);
    //         std::cout << "done." << std::endl;
    //         timer.measure("Saving");
    //     }

    //     std::cout << "Whole texturing procedure took: " << wtimer.get_elapsed_sec() << "s" << std::endl;
    //     timer.measure("Total");
    //     if (conf.write_timings) {
    //         timer.write_to_file(conf.out_prefix + "_timings.csv");
    //     }

    //     if (conf.write_view_selection_model) {
    //         texture_atlases.clear();
    //         std::cout << "Generating debug texture patches:" << std::endl;
    //         {
    //             tex::TexturePatches texture_patches;
    //             generate_debug_embeddings(&image_views);
    //             tex::VertexProjectionInfos vertex_projection_infos; // Will only be written
    //             tex::generate_texture_patches(graph, mesh, mesh_info, &image_views,
    //                 conf.settings, &vertex_projection_infos, &texture_patches);
    //             tex::generate_texture_atlases(&texture_patches, conf.settings, &texture_atlases, type, false);
    //         }

    //         std::cout << "Building debug objmodel:" << std::endl;
    //         {
    //             tex::Model model;
    //             tex::build_model(mesh, texture_atlases, &model);
    //             std::cout << "\tSaving model... " << std::flush;
    //             tex::Model::save(model, conf.out_prefix + "_view_selection");
    //             std::cout << "done." << std::endl;
    //         }
    //     }

    //     /* Remove temporary files. */
    //     for (util::fs::File const & file : util::fs::Directory(tmp_dir)) {
    //         util::fs::unlink(util::fs::join_path(file.path, file.name).c_str());
    //     }
    //     util::fs::rmdir(tmp_dir.c_str());

    return EXIT_SUCCESS;
}
