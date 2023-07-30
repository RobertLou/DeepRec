#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <filesystem>

int main() {
    // 指定要统计的文件夹路径
    std::string folderPath = "/root/code/DeepRec/time";

    double miss_num = 0.0, gather_time = 0.0, miss_time = 0.0, get_value_address_time = 0.0, total_time = 0.0;
    double HBMlookup_time = 0.0, DRAMlookup_time = 0.0, copyToOutput_time = 0.0, LookupAll_time = 0.0;
    // 遍历文件夹中的每个文件
    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        double totalSum = 0.0;
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            std::ifstream inputFile(entry.path().string());
            if (inputFile.is_open()) {
                std::string line;
                while (std::getline(inputFile, line)) {
                    // 使用字符串流从文本行中提取数据
                    double data;
                    std::istringstream iss(line);
                    if (iss >> data) {
                        totalSum += data;
                    }
                }
                inputFile.close();
            } else {
                std::cerr << "无法打开文件：" << entry.path() << std::endl;
            }

            if(entry.path().string().find("gather") != std::string::npos){
                gather_time += totalSum;
            }
            else if(entry.path().string().find("getValueAddress") != std::string::npos){
                get_value_address_time += totalSum;
            }
            else if(entry.path().string().find("getmissing") != std::string::npos){
                miss_time += totalSum;
            }
            else if(entry.path().string().find("missnum") != std::string::npos){
                miss_num += totalSum;
            }
            else if(entry.path().string().find("total") != std::string::npos){
                total_time += totalSum;
            }
            else if(entry.path().string().find("HBMlookup") != std::string::npos){
                HBMlookup_time += totalSum;
            }
            else if(entry.path().string().find("DRAMlookup") != std::string::npos){
                DRAMlookup_time += totalSum;
            }
            else if(entry.path().string().find("copyToOutput") != std::string::npos){
                copyToOutput_time += totalSum;
            }
            else if(entry.path().string().find("LookupAll") != std::string::npos){
                LookupAll_time += totalSum;
            }
            std::cout << entry.path().string() << "文件中的数据总和为：" << totalSum << std::endl;
        }
    }
    std::cout << "gather_time: " << gather_time << std::endl;
    std::cout << "get_value_address_time: " << get_value_address_time << std::endl;
    std::cout << "miss_time: " << miss_time << std::endl;
    std::cout << "miss_num: " << miss_num << std::endl;
    std::cout << "miss_rate: " << miss_num / 26557961 << std::endl;
    std::cout << "total_time: " << total_time << std::endl;
    std::cout << "HBMlookup_time: " << HBMlookup_time << std::endl;
    std::cout << "DRAMlookup_time: " << DRAMlookup_time << std::endl;
    std::cout << "copyToOutput_time: " << copyToOutput_time << std::endl;
    std::cout << "LookupAll_time: " << LookupAll_time << std::endl;

    return 0;
}