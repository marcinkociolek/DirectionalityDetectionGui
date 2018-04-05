#include "dirdetectionparams.h"

using namespace std;

DirDetectionParams::DirDetectionParams()
{
    DefaultParams();
}

void DirDetectionParams::DefaultParams(void)
{
    InFolderName = "";
    InFilePattern = ".+";

    OutFolderName1 = "";
    OutFolderName2 = "";

    preprocessType = 0;
    preprocessKernelSize = 3;

    showInputGray = false;
    showInputPC = true;
    showRoi = true;
    showSmallImage = true;

    tileShape = 2;

    tileSize = 61;
    //tileWidth = 61;
    //tileHeight = 61;
    tileShift = 45;
    //tileShiftY = 45;
    tileOffsetX = 31;
    tileOffsetY = 31;

    showTiles = true;
    tileLineWidth = 1;

    textOut = true;
    imgOut = false;

    normalisation = 0;

    binCount = 16;

    angleStep = 1.0;

    minOffset = 8;
    offsetCount = 4;
    offsetStep = 1;

    fixMinNorm;
    fixMaxNorm ;

    displayGrayMax = 40000;
    displayGrayMin = 0;

    displayPCMax = 65535;
    displayPCMin = 0;

    showDirection = true;
    directionLineWidth = 2;
    directionLineLength = 23;

    showOutputText = true;
    calculateDirectionality = false;
}

std::string DirDetectionParams::ShowParams()
{
    string OutString = "";

    OutString += "Input Directory 1:\t"		+ InFolderName		+ "\n";
    OutString += "Input File Name Base 1:\t"	+ InFilePattern	+ "\n";


    OutString += "Output Directory:\t" + OutFolderName1 + "\n";
    OutString += "Output Directory 2:\t" + OutFolderName2 + "\n";

    OutString += "Preprocess Type:\t";
    OutString += to_string(preprocessType);
    OutString += " - ";
    switch (preprocessType)
    {
    case 1:
        OutString += "average blurr";
        break;
    case 2:
        OutString += "median blurr";
        break;
    default:
        OutString += "none";
        break;
    }
    OutString += "\n";

    OutString += "Preprocess kernel size:\t";
    OutString += to_string(preprocessKernelSize);
    OutString += "\n";

    OutString += "Display result in gray:\t";
    if (showInputGray)
        OutString += "Y";
    else
        OutString += "N";
    OutString += "\n";

    OutString += "Display result in pseudocolor:\t";
    if (showInputPC)
        OutString += "Y";
    else
        OutString += "N";
    OutString += "\n";


    OutString += "Display small image:\t";
    if (showSmallImage)
        OutString += "Y";
    else
        OutString += "N";
    OutString += "\n";

    OutString += "Tile Shape:\t";
    OutString += to_string(tileShape);
    OutString += " - ";
    switch (tileShape)
    {
    case 1:
        OutString += "Rectangular";
        break;
    case 2:
        OutString += "Elyptical";
        break;
    case 3:
        OutString += "Hexagonal";
        break;
    default:
        OutString += "none";
        break;
    }
    OutString += "\n";

    OutString += "Tile width x:\t";
    OutString += to_string(tileSize);
    OutString += "\n";

    OutString += "Tile shift:\t";
    OutString += to_string(tileShift);
    OutString += "\n";

    //OutString += "Tile shift y:\t";
    //OutString += to_string(tileShiftY);
    //OutString += "\n";

    OutString += "Tile offset x:\t";
    OutString += to_string(tileOffsetX);
    OutString += "\n";

    OutString += "Tile offset y:\t";
    OutString += to_string(tileOffsetY);
    OutString += "\n";


    OutString += "Show tiles:\t";
    if (showTiles)
        OutString += "Y";
    else
        OutString += "N";
    OutString += "\n";

    OutString += "Tile Line Width>:\t";
    OutString += to_string(tileLineWidth);
    OutString += "\n";

    OutString += "Text out:\t";
    if (textOut)
        OutString += "Y";
    else
        OutString += "N";
    OutString += "\n";

    OutString += "Image out:\t";
    if (imgOut)
        OutString += "Y";
    else
        OutString += "N";
    OutString += "\n";

    OutString += "Normalisation:\t";
    OutString += to_string(normalisation);
    OutString += " - ";
    switch (normalisation)
    {
    case 1:
        OutString += "Tile min - max";
        break;
    case 2:
        OutString += "Global min - max";
        break;
    case 3:
        OutString += "Tile +/- 3 sigma";
        break;
    case 4:
        OutString += "Global +/- 3 sigma";
        break;
    case 5:
        OutString += "Tile 1%-99%";
        break;
    case 6:
        OutString += "Global 1%-99%";
        break;
    default:
        OutString += "none";
        break;
    }
    OutString += "\n";

    OutString += "Bin count:\t";
    OutString += to_string(binCount);
    OutString += "\n";

    OutString += "Min Offset:\t";
    OutString += to_string(minOffset);
    OutString += "\n";

    OutString += "Offset count:\t";
    OutString += to_string(offsetCount);
    OutString += "\n";

    OutString += "Offset step:\t";
    OutString += to_string(offsetStep);
    OutString += "\n";


    OutString += "Fixed Min Norm:\t";
    OutString += to_string(fixMinNorm);
    OutString += "\n";

    OutString += "Fixed Max Norm:\t";
    OutString += to_string(fixMaxNorm);
    OutString += "\n";

    OutString += "Display Max Gray:\t";
    OutString += to_string(displayGrayMax);
    OutString += "\n";

    OutString += "Display Min Gray:\t";
    OutString += to_string(displayGrayMin);
    OutString += "\n";

    OutString += "Display Max pseudocolor:\t";
    OutString += to_string(displayPCMax);
    OutString += "\n";

    OutString += "Display Min pseudocolor:\t";
    OutString += to_string(displayPCMin);
    OutString += "\n";

    OutString += "Show Direction on image:\t";
    if (showDirection)
        OutString += "Y";
    else
        OutString += "N";
    OutString += "\n";

    OutString += "Direction Line Width>:\t";
    OutString += to_string(directionLineWidth);
    OutString += "\n";

    OutString += "Durection Line Length:\t";
    OutString += to_string(directionLineLength);
    OutString += "\n";

    OutString += "Angle Step:\t";
    OutString += to_string(angleStep);
    OutString += "\n";

    OutString += "\n";

    return OutString;
}
