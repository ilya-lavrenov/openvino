// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief The entry point for inference engine deconvolution sample application
 * @file style_transfer_sample/main.cpp
 * @example style_transfer_sample/main.cpp
 */
#include <vector>
#include <string>
#include <memory>

#include <format_reader_ptr.h>
#include <inference_engine.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "style_transfer_sample.h"

using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    slog::info << "Parsing input parameters" << slog::endl;

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

void debugImageOutput(const cv::Mat& inputImage, const std::string fileName)
{
    cv::Mat tempImage = inputImage.clone();
    cv::cvtColor(inputImage, tempImage, cv::COLOR_RGB2BGR);
    cv::imwrite("nnart" + fileName + ".jpg", inputImage);
    cv::cvtColor(inputImage, tempImage, cv::COLOR_BGR2RGB);
}

cv::Mat prepareImage(const cv::Mat& rawImage, int maxSize)
{
    constexpr int fixedBufferSize = 1024;
    debugImageOutput(rawImage, "1_original");

    cv::Mat preparedImage;
    preparedImage.create(fixedBufferSize, fixedBufferSize, rawImage.type());
    preparedImage.setTo(cv::Scalar::all(0));

    const float scaling_factor = std::min(
        maxSize / static_cast<float>(rawImage.rows),
        maxSize / static_cast<float>(rawImage.cols)
    );

    cv::Mat scalledImage;
    cv::resize(rawImage, scalledImage, cv::Size(0, 0), scaling_factor, scaling_factor, cv::INTER_LANCZOS4);
    debugImageOutput(scalledImage, "2_resized_original");
    scalledImage.copyTo(preparedImage(cv::Rect(0, 0, scalledImage.cols, scalledImage.rows)));

    preparedImage.convertTo(preparedImage, CV_32FC3);
    debugImageOutput(preparedImage, "3_inset_original");

    return preparedImage;
}

void postProcessImage(cv::Mat& dstImage, const cv::Mat& rawImage, int maxSize)
{
    int newOutSizes[3] = { (int)dstImage.rows, (int)dstImage.cols, 3 };

    cv::Mat m = cv::Mat(3, newOutSizes, CV_32FC1, dstImage.data);

    //TODO: Its possible to move this conversion to model itself.
    m.forEach<float>([&](float& element, const int position[]) -> void {
        element += 1.0f;
        element /= 2.0f;
        element *= 255.0f;
        element = element >= 255 ? 255 : element;
        element = element <= 0 ? 0 : element;
        });

    dstImage.convertTo(dstImage, CV_8UC3);
    debugImageOutput(dstImage, "4_processed");

    const float scaling_factor = std::min(
        maxSize / static_cast<float>(rawImage.rows),
        maxSize / static_cast<float>(rawImage.cols)
    );
    const cv::Rect cropRect(0, 0, rawImage.cols * scaling_factor, rawImage.rows * scaling_factor);
    dstImage = dstImage(cropRect);
    debugImageOutput(dstImage, "5_processed_cropped");

    cv::resize(dstImage, dstImage, cv::Size(rawImage.cols, rawImage.rows), 0, 0, cv::INTER_LANCZOS4);
    debugImageOutput(dstImage, "6_processed_resized");
}

int main(int argc, char *argv[]) {
    try {
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;
        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** This vector stores paths to the processed images **/
        std::vector<std::string> imageNames;
        parseInputFilesArguments(imageNames);
        if (imageNames.empty()) throw std::logic_error("No suitable images were found");
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load inference engine -------------------------------------
        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;

        /** Printing device version **/
        slog::info << "Device info: " << slog::endl;
        std::cout << ie.GetVersions(FLAGS_d) << std::endl;

        if (!FLAGS_l.empty()) {
            // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
            ie.AddExtension(extension_ptr);
            slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
        }
        if (!FLAGS_c.empty()) {
            // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
            ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
            slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
        }
        // -----------------------------------------------------------------------------------------------------

        // 2. Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
        slog::info << "Loading network files" << slog::endl;

        /** Read network model **/
        CNNNetwork network = ie.ReadNetwork(FLAGS_m);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------

        // --------------------------- Prepare input blobs -----------------------------------------------------
        slog::info << "Preparing input blobs" << slog::endl;

        /** Taking information about all topology inputs **/
        InputsDataMap inputInfo(network.getInputsInfo());

        if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");
        auto inputInfoItem = *inputInfo.begin();
        std::string input_name = network.getInputsInfo().begin()->first;

        /** Iterate over all the input blobs **/
        std::vector<std::shared_ptr<uint8_t>> imagesData;
        //std::vector<std::shared_ptr<float>> imagesData;

        /** Specifying the precision of input data.
         * This should be called before load of the network to the device **/
        inputInfoItem.second->setLayout(Layout::NHWC);
        inputInfoItem.second->setPrecision(Precision::FP32);
        inputInfoItem.second->setNetworkLayout(NetworkLayout(NHWC));

        //FormatReader::ReaderPtr reader(imageNames);
        cv::Mat rawImage;
        cv::Mat fImage;

        //int newInSizes[3] = { 1024, 1024, 3 };
        //rawImage = cv::Mat(3, newInSizes, CV_8UC1, NULL);
        rawImage = cv::Mat(1024, 1024, CV_8UC3);

        /** Collect images data ptrs **/
        for (auto & i : imageNames) {
            FormatReader::ReaderPtr reader(i.c_str());
            //tmpImage = cv::Mat((int)image_height, (int)image_width, CV_8UC3);
            if (reader.get() == nullptr) {
                slog::warn << "Image " + i + " cannot be read!" << slog::endl;
                continue;
            }
            /** Store image data **/
            size_t channels = inputInfoItem.second->getTensorDesc().getDims()[3];
            size_t image_h = inputInfoItem.second->getTensorDesc().getDims()[1];
            size_t image_w = inputInfoItem.second->getTensorDesc().getDims()[2];

            std::shared_ptr<unsigned char> data(reader->getData(image_w, image_h));
            //std::shared_ptr<float> data(reader->getData(image_w, image_h));

            if (data.get() != nullptr) {
                imagesData.push_back(data);
                std::memcpy((void*)rawImage.data, data.get(), sizeof(char) * 1024 * 1024 * 3);
                fImage = prepareImage(rawImage, 1024);
            }
        }
        if (imagesData.empty()) throw std::logic_error("Valid input images were not found!");

        /** Setting batch size using image count **/
        network.setBatchSize(imagesData.size());
        slog::info << "Batch size is " << std::to_string(network.getBatchSize()) << slog::endl;

        // ------------------------------ Prepare output blobs -------------------------------------------------
        slog::info << "Preparing output blobs" << slog::endl;

        OutputsDataMap outputInfo(network.getOutputsInfo());
        // BlobMap outputBlobs;
        std::string firstOutputName;

        const float meanValues[] = {static_cast<const float>(FLAGS_mean_val_r),
                                    static_cast<const float>(FLAGS_mean_val_g),
                                    static_cast<const float>(FLAGS_mean_val_b)};

        for (auto & item : outputInfo) {
            if (firstOutputName.empty()) {
                firstOutputName = item.first;
            }
            DataPtr outputData = item.second;
            if (!outputData) {
                throw std::logic_error("output data pointer is not valid");
            }
            // item.second->setLayout(Layout::NHWC);
            item.second->setPrecision(Precision::FP32);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the device ------------------------------------------
        slog::info << "Loading model to the device" << slog::endl;
        ExecutableNetwork executable_network = ie.LoadNetwork(network, FLAGS_d);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        slog::info << "Create infer request" << slog::endl;
        InferRequest infer_request = executable_network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Prepare input --------------------------------------------------------
        /** Iterate over all the input blobs **/
        for (const auto & item : inputInfo) {
            MemoryBlob::Ptr minput = as<MemoryBlob>(infer_request.GetBlob(input_name));
            if (!minput) {
                slog::err << "We expect input blob to be inherited from MemoryBlob, " <<
                    "but by fact we were not able to cast it to MemoryBlob" << slog::endl;
                return 1;
            }
            // locked memory holder should be alive all time while access to its buffer happens
            auto ilmHolder = minput->wmap();

            /** Filling input tensor with images. First b channel, then g and r channels **/
            size_t num_channels = minput->getTensorDesc().getDims()[3];
            size_t image_size = minput->getTensorDesc().getDims()[2] * minput->getTensorDesc().getDims()[1] * 3;

            auto data = ilmHolder.as<PrecisionTrait<Precision::FP32>::value_type *>();
            float* fdata = (float*)fImage.data;

            /** Iterate over all input images **/
            for (size_t image_id = 0; image_id < imagesData.size(); ++image_id) {
                /** Iterate over all pixel in image (b,g,r) **/
                //for (size_t pid = 0; pid < image_size; pid++) {
                for (size_t pid = 0; pid < image_size; pid++) {
                    data[pid] = fdata[pid];
                    /** Iterate over all channels **/
                    //for (size_t ch = 0; ch < num_channels; ++ch) {
                        /**          [images stride + channels stride + pixel id ] all in bytes            **/
                    //    data[image_id * image_size * num_channels + ch * image_size + pid ] =
                    //        imagesData.at(image_id).get()[pid*num_channels + ch] - meanValues[ch];
                    //    slog::info << "Processing... data[" << image_id * image_size* num_channels + ch * image_size + pid << "] from imgData[" << pid * num_channels + ch << "]." << slog::endl;
                    //}
                }
            }
            cv::Mat inImage = cv::Mat(1024, 1024, CV_32FC3);
            std::memcpy((void*)inImage.data, (void*)(data), sizeof(float) * 1024 * 1024 * 3);
            //prepareImage(inImage, 1024);
            infer_request.SetBlob(input_name, minput);  // infer_request accepts input blob of any size
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Do inference ---------------------------------------------------------
        slog::info << "Start inference" << slog::endl;
        infer_request.Infer();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Process output -------------------------------------------------------
        MemoryBlob::CPtr moutput = as<MemoryBlob>(infer_request.GetBlob(firstOutputName));
        if (!moutput) {
            throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                                   "but by fact we were not able to cast it to MemoryBlob");
        }
        // locked memory holder should be alive all time while access to its buffer happens
        auto lmoHolder = moutput->rmap();
        const auto output_data = lmoHolder.as<const PrecisionTrait<Precision::FP32>::value_type *>();

        size_t num_images = moutput->getTensorDesc().getDims()[0];
        size_t num_channels = moutput->getTensorDesc().getDims()[3];
        size_t H = moutput->getTensorDesc().getDims()[1];
        size_t W = moutput->getTensorDesc().getDims()[2];
        size_t nPixels = W * H;

        slog::info << "Output size [N,C,H,W]: " << num_images << ", " << num_channels << ", " << H << ", " << W << slog::endl;

        {
            std::vector<float> data_img(nPixels * num_channels);
            std::vector<float> data_img2(nPixels * num_channels);

            for (int n = 0; n < num_images; n++) {
                for (int i = 0; i < nPixels * 3; i++) {

                    data_img[i] = output_data[i];
                    data_img2[i] = output_data[i];

                    data_img[i] += 1.0f;
                    data_img[i] /= 2.0f;
                    data_img[i] *= 255.0f;

                    if (data_img[i] < 0.0f) data_img[i] = 0.0f;
                    if (data_img[i] > 255.0f) data_img[i] = 255.0f;
                }

                cv::Mat dstImage = cv::Mat((int)H, (int)W, CV_32FC3);
                std::memcpy((void*)dstImage.data, (void *)(data_img2.data()), sizeof(float) * H * W * 3);
                postProcessImage(dstImage, rawImage, 1024);

                std::string out_img_name = std::string("out" + std::to_string(n + 1) + ".bmp");
                std::ofstream outFile;
                outFile.open(out_img_name.c_str(), std::ios_base::binary);
                if (!outFile.is_open()) {
                    throw new std::runtime_error("Cannot create " + out_img_name);
                }
                std::vector<unsigned char> data_img2;
                for (float i : data_img) {
                    data_img2.push_back(static_cast<unsigned char>(i));
                }
                writeOutputBmp(data_img2.data(), H, W, outFile);
                outFile.close();
                slog::info << "Image " << out_img_name << " created!" << slog::endl;
            }
        }
        // -----------------------------------------------------------------------------------------------------
    }
    catch (const std::exception &error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened" << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    slog::info << slog::endl << "This sample is an API example, for any performance measurements "
                                "please use the dedicated benchmark_app tool" << slog::endl;
    return 0;
}