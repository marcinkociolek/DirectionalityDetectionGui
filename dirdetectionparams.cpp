#include "dirdetectionparams.h"

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
    showSmallImage = true;

    tileShape = 2;

    tileWidth = 61;
    tileHeight = 61;
    tileShiftX = 45;
    tileShiftY = 45;
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
    offsetCount = 6;
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


}
