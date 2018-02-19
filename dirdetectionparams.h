#ifndef DIRDETECTIONPARAMS_H
#define DIRDETECTIONPARAMS_H

#include <string>


class DirDetectionParams
{
public:

    std::string InFolderName;
    std::string InFilePattern;

    std::string OutFolderName1;
    std::string OutFolderName2;

    int preprocessType;
    int preprocessKernelSize;

    bool showInputGray;
    bool showInputPC;
    bool showSmallImage;

    int tileShape;

    int tileWidth;
    int tileHeight;
    int tileShiftX;
    int tileShiftY;
    int tileOffsetX;
    int tileOffsetY;

    bool showTiles;
    int tileLineWidth;

    bool textOut;
    bool imgOut;

    int normalisation;

    int binCount;

    int minOffset;
    int offsetCount;
    int offsetStep;

    float fixMinNorm;
    float fixMaxNorm ;

    float displayGrayMax;
    float displayGrayMin;

    float displayPCMax;
    float displayPCMin;

    bool showDirection;
    int directionLineWidth;
    int directionLineLength;

    float angleStep;

    // functions
    DirDetectionParams();
    void DefaultParams(void);
    //int LoadParams(string XmlFileName);
    //int SaveParams(string XmlFileName);
    std::string ShowParams(void);
};

#endif // DIRDETECTIONPARAMS_H
