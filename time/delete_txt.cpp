#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

void deleteTxtFiles(const std::string& folderPath) {
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            if (fs::remove(entry.path())) {
                std::cout << "已删除文件：" << entry.path() << std::endl;
            } else {
                std::cerr << "无法删除文件：" << entry.path() << std::endl;
            }
        }
    }
}

int main() {
    std::string folderPath = "/root/code/DeepRec/time"; // 替换为实际文件夹路径
    deleteTxtFiles(folderPath);

    return 0;
}