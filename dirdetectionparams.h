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

    std::string FileName; //this is only for passing the finamame to the procedude not to store in xmlfile

    int preprocessType;
    int preprocessKernelSize;

    bool showInputGray;
    bool showInputPC;
    bool showRoi;
    bool showSmallImage;

    int tileShape;

    int tileSize;
    //int tileHeight;
    int tileShift;
    //int tileShiftY;
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

    double angleStep;

    float displayGrayMax;
    float displayGrayMin;

    float displayPCMax;
    float displayPCMin;

    bool showDirection;
    int directionLineWidth;
    int directionLineLength;

    bool showOutputText;
    bool calculateDirectionality;

    // functions
    DirDetectionParams();
    void DefaultParams(void);
    //int LoadParams(string XmlFileName);
    //int SaveParams(string XmlFileName);
    std::string ShowParams(void);
};

#endif // DIRDETECTIONPARAMS_H
