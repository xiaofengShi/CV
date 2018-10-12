/*
ayoungprogrammer.blogspot.com

Part 2: Generating Tesseract data

*/

#include <iostream>
#include <fstream>
#include <windows.h>
#include "cv.h"
#include "highgui.h"

using namespace std;
using namespace cv;

int numOfFiles(char *searchPath)
{
    WIN32_FIND_DATA FindData;
    HANDLE hFiles;
    LPTSTR lptszFiles[100];
    UINT nFileCount = 0;

    hFiles = FindFirstFile(searchPath, &FindData);

    if (hFiles == INVALID_HANDLE_VALUE)
        return 0;

    bool bFinished = false;
    while (!bFinished)
    {
        if (!(FindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
        {
            nFileCount++;
        }

        if (!FindNextFile(hFiles, &FindData))
        {
            bFinished = true;
        }
    }
    FindClose(hFiles);
    return nFileCount;
}

int main(void)
{

    string fileName = "mat.arial.exp0";
    cout << "Output name: ";

    ofstream box("mat.arial.exp0.box");

    char cCurrentPath[256];
    GetCurrentDirectory(sizeof(cCurrentPath), cCurrentPath);

    string keys = "()+0x123456789";
    Mat tiff = Mat::zeros(1800, 1200, CV_8UC3);
    Mat tiff2 = Mat::zeros(1800, 1200, CV_8UC3);
    for (int j = 0; j < keys.length(); j++)
    {

        stringstream fileOutput;
        if (keys[j] == '|')
            fileOutput << cCurrentPath << "\\output\\"
                       << "abs"
                       << "\\";
        else
            fileOutput << cCurrentPath << "\\output\\" << keys[j] << "\\";

        stringstream fileSearch;
        fileSearch << fileOutput.str();
        fileSearch << "*.jpg";

        cout << fileSearch.str() << endl;
        int n = numOfFiles((char *)fileSearch.str().c_str());
        cout << n << endl;
        float targetHeight = 50;
        int colDist = 0;
        for (int i = 0; i < min(13, n); i++)
        {
            stringstream filePath;
            filePath << fileOutput.str();
            filePath << i << ".jpg";
            Mat symbol = imread(filePath.str());

            float percent = targetHeight / symbol.size().height;
            if (percent * symbol.size().width > 50)
            {
                percent = targetHeight / symbol.size().width;
            }
            Mat resized;
            resize(symbol, resized, Size(), percent, percent);
            Mat small = tiff.colRange(75 * i + 50, 50 + 75 * i + resized.size().width).rowRange(75 * j + 50, 75 * j + resized.size().height + 50);

            resized.copyTo(small);
            box << keys[j] << " " << 75 * i + 50 << " " << tiff.size().height - (75 * j + resized.size().height + 50) << " "
                << 50 + 75 * i + resized.size().width << " " << tiff.size().height - (75 * j + 50) << " 0" << endl;

            Rect r = Rect(75 * i + 50, 75 * j + 50, resized.size().width, resized.size().height);
            Mat small2 = tiff2.colRange(75 * i + 50, 50 + 75 * i + resized.size().width).rowRange(75 * j + 50, 75 * j + resized.size().height + 50);
            resized.copyTo(small2);
            rectangle(tiff2, r.tl(), r.br(), Scalar(0, 255, 0), 2, 8, 0);
        }
    }
    imwrite(fileName + ".tif", tiff);
    box.close();

    waitKey(0);
}