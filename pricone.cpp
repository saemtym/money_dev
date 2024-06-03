#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <filesystem>

int main() {
    // OpenCLを有効にする
    cv::ocl::setUseOpenCL(true);

    // 大きな画像を読み込む
    cv::UMat large_image = cv::imread("decks/IMG_3309.png", cv::IMREAD_GRAYSCALE).getUMat(cv::ACCESS_READ);
    cv::Mat large_image_host;

    // charsディレクトリ内のすべてのpngファイルに対してテンプレートマッチングを行う
    for (const auto & entry : std::filesystem::directory_iterator("chars")) {
        if (entry.path().extension() == ".png") {
            // テンプレート画像を読み込む
            cv::UMat templateImage = cv::imread(entry.path(), cv::IMREAD_GRAYSCALE).getUMat(cv::ACCESS_READ);
            int w = templateImage.cols;
            int h = templateImage.rows;

            // テンプレートマッチングを行う
            cv::UMat res;
            cv::matchTemplate(large_image, templateImage, res, cv::TM_CCOEFF_NORMED);
            double threshold = 0.8;

            // マッチした領域を元の画像に描画する
            cv::UMat loc;
            cv::threshold(res, loc, threshold, 1., cv::THRESH_BINARY);
            cv::Mat loc_host = loc.getMat(cv::ACCESS_READ);
            large_image_host = large_image.getMat(cv::ACCESS_RW);
            for (int y = 0; y < loc_host.rows; y++) {
                for (int x = 0; x < loc_host.cols; x++) {
                    if (loc_host.at<unsigned char>(y, x)) {
                        cv::rectangle(large_image_host, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(0, 0, 255), 2);
                    }
                }
            }
        }
    }

    // 結果を表示する
    cv::imshow("Detected", large_image_host);
    cv::waitKey(0);

    return 0;
}