#include "mainwindow.h"
#include "ui_mainwindow.h"


#include <QFileDialog>
#include <QMessageBox>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <math.h>
#include <iostream>
#include <fstream>
#include <ctime>

#include "DispLib.h"
#include "NormalizationLib.h"
#include "HaralickLib.h"
#include "displayfordirdetection.h"

#define PI 3.14159265

using namespace std;
using namespace boost::filesystem;
using namespace boost;
using namespace cv;
//-----------------------------------------------------------------------------------------------
// Directionality calculation functions
//-----------------------------------------------------------------------------------------------
cv::Mat CreateStandardROI(int size, int roiShape)
{
    Mat Roi;
    switch (roiShape) // Different tile shapes
    {
    case 1: // Circle
        Roi = Mat::zeros(size, size, CV_16U);
        ellipse(Roi, Point(size / 2, size / 2),	Size(size / 2, size / 2), 0.0, 0.0, 360.0, 1, -1);
        break;
    case 2: // Hexagon
    {
        int edgeLength = size/2;
        int roiMaxX = size;
        int roiMaxY = (int)((double)size * 0.8660254);
        Roi = Mat::zeros(roiMaxY, roiMaxX, CV_16U);

        Point vertice0(edgeLength / 2, 0);
        Point vertice1(edgeLength / 2 + edgeLength - 1, 0);
        Point vertice2(roiMaxX - 1, roiMaxY / 2);
        Point vertice3(edgeLength / 2 + edgeLength - 1, roiMaxY - 1);
        Point vertice4(edgeLength / 2, roiMaxY - 1);
        Point vertice5(0, roiMaxY / 2);

        line(Roi, vertice0, vertice1, 1, 1);
        line(Roi, vertice1, vertice2, 1, 1);
        line(Roi, vertice2, vertice3, 1, 1);
        line(Roi, vertice3, vertice4, 1, 1);
        line(Roi, vertice4, vertice5, 1, 1);
        line(Roi, vertice5, vertice0, 1, 1);
        unsigned short *wRoi;

        for (int y = 1; y < roiMaxY - 1; y++)
        {
            wRoi = (unsigned short *)Roi.data + roiMaxX * y;
            int x = 0;
            for (x; x < roiMaxX; x++)
            {
                if (*wRoi)
                    break;
                wRoi++;
            }
            x++;
            wRoi++;
            for (x; x < roiMaxX; x++)
            {
                if (*wRoi)
                    break;
                *wRoi = 1;
                wRoi++;
            }
        }
    }
    break;
    default:// square
        Roi = Mat::ones(size, size, CV_16U);
        break;
    }
    return Roi;
}
//-----------------------------------------------------------------------------------------------
void Preprocess(cv::Mat Im, int preprocessType, int preprocesKernelSize)
{
    switch (preprocessType)
    {
    case 1:
        blur(Im, Im, Size(preprocesKernelSize, preprocesKernelSize));
        break;
    case 2:
        medianBlur(Im, Im, preprocesKernelSize);
        break;
    default:
        break;
    }
}
//-----------------------------------------------------------------------------------------------
void GlobalNormalisation(cv::Mat ImF, int normalisation, float *maxNormGlobal, float *minNormGlobal)
{
    switch (normalisation)
    {
    case 2:
        NormParamsMinMax(ImF, maxNormGlobal, minNormGlobal);
        break;
    case 4:
        NormParamsMeanP3Std(ImF, maxNormGlobal, minNormGlobal);
        break;
    case 6:
        NormParams1to99perc(ImF, maxNormGlobal, minNormGlobal);
        break;
    default:
        *minNormGlobal = 0.0;
        *maxNormGlobal = 65535.0;
        break;
    }
}

//-----------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------
// my functions
//-----------------------------------------------------------------------------------------------

void ImShowPC(cv::Mat ImIn,  DirDetectionParams params)
{
    if(!params.showInputPC)
        return;
    if(ImIn.empty())
        return;
    Mat ImToShow = ShowImage16PseudoColor(ImIn, params.displayPCMin, params.displayPCMax);
    DrawTilesOnImage(ImToShow, params);
    if (params.imagesScale !=1.0)
        cv::resize(ImToShow,ImToShow,Size(),params.imagesScale,params.imagesScale,INTER_NEAREST);
    imshow("Input Pseudo Color", ImToShow);
    ImToShow.release();
}
//-----------------------------------------------------------------------------------------------
void ImShowGray(cv::Mat ImIn,  DirDetectionParams params)
{
    if(!params.showInputGray)
        return;
    if(ImIn.empty())
        return;
    Mat ImToShow = ShowImage16Gray(ImIn, params.displayGrayMin, params.displayGrayMax);
    DrawTilesOnImage(ImToShow, params);
    if (params.imagesScale !=1.0)
        cv::resize(ImToShow,ImToShow,Size(),params.imagesScale,params.imagesScale,INTER_NEAREST);
    imshow("Input Gray", ImToShow);
    ImToShow.release();
}
//-----------------------------------------------------------------------------------------------
Mat PrepareImShow(cv::Mat ImIn,  DirDetectionParams params)
{
    Mat ImToShow = ShowImage16PseudoColor(ImIn, params.displayPCMin, params.displayPCMax);
    DrawTilesOnImage(ImToShow, params);
    return ImToShow;

}

//-----------------------------------------------------------------------------------------------
void ShowDirectionSmall(Mat ImIn, float direction, DirDetectionParams params, float minNorm, float maxNorm)
{
    Mat ImToShow = ShowImageF32PseudoColor(ImIn, minNorm, maxNorm);
    int lineOffsetX = (int)round(params.directionLineLength * 0.5 *  sin((double)direction* PI / 180.0));
    int lineOffsetY = (int)round(params.directionLineLength * 0.5 * cos((double)direction* PI / 180.0));

    line(ImToShow, Point(params.tileOffsetX - lineOffsetX, params.tileOffsetY - lineOffsetY), Point(params.tileOffsetX + lineOffsetX, params.tileOffsetY + lineOffsetY), Scalar(0, 0.0, 0.0, 0.0), params.directionLineWidth);

    imshow("ImSmall", ImToShow);

}

//-----------------------------------------------------------------------------------------------
string DirEstimation(cv::Mat ImIn,  DirDetectionParams params, bool *stopCalc)
{
    Mat Roi = CreateStandardROI(params.tileSize, params.tileShape);
    if(params.showTileRoiImage)
        imshow("Roi", Roi*65535);
    if(ImIn.empty())
        return "Error 1 Empty Image";

    int stepNr = (int)(180.0 / params.angleStep); // angle step for computations (number of steps)

    float *CorrelationAvg = new float[stepNr];

    //Matrix declarations
    Mat ImInF, ImToShow, SmallIm, COM, SmallImToShow;

    if(params.showOutputImage)
        ImToShow = PrepareImShow(ImIn,params);

    ImIn.convertTo(ImInF, CV_32F);

    int maxX = ImInF.cols;
    int maxY = ImInF.rows;

    float maxNormGlobal = 65535;
    float minNormGlobal = 0;

    GlobalNormalisation(ImInF, params.normalisation , &maxNormGlobal, &minNormGlobal);

    string OutDataString = params.ShowParams();
    OutDataString += "Tile Y\tTile X\t";
    OutDataString += "Angle \t";
    OutDataString += "Tile min norm\tTile max norm\t";
    OutDataString += "\n";

    int maxOffset = params.minOffset + params. offsetCount * params.offsetStep;

    int firstTileY = params.tileOffsetY;
    int lastTileY = maxY - params.tileSize / 2;
    int firstTileX = params.tileOffsetX;
    int lastTileX = maxX - params.tileSize / 2;
    for (int y = firstTileY; y <= lastTileY; y += params.tileShift)
    {
        for (int x = firstTileX; x <= lastTileX; x += params.tileShift)
        {

            ImInF(Rect(x - Roi.cols / 2, y - Roi.rows / 2, Roi.cols, Roi.rows)).copyTo(SmallIm);
            float maxNorm, minNorm;
            switch (params.normalisation)
            {
            case 1:
                NormParamsMinMax(SmallIm, Roi, 1, &maxNorm, &minNorm);
                break;
            case 3:
                NormParamsMeanP3Std(SmallIm, Roi, 1, &maxNorm, &minNorm);
                break;
            case 5:
                NormParams1to99perc(SmallIm, Roi, 1, &maxNorm, &minNorm);
                break;
            default:
                maxNorm = maxNormGlobal;
                minNorm = minNormGlobal;
                break;
            }

            for (int i = 0; i < stepNr; i++)
            {
                //Angles[i] = 0;
                CorrelationAvg[i] = 0;
            }
            int bestAngleCorAvg;

            // ofset loop--------------------------------------------------------------------------
            for (int offset = params.minOffset; offset <= maxOffset; offset += params.offsetStep)
            {
                for (int angleIndex = 0; angleIndex < stepNr; angleIndex++)
                {
                    float angle = params.angleStep * angleIndex;

                    COM.release();

                    if (params.tileShape)
                        COM = COMCardoneRoi(SmallIm, Roi, offset, angle, params.binCount, maxNorm, minNorm, 0, 1);
                    else
                        COM = COMCardone4(SmallIm, offset, angle, params.binCount, maxNorm, minNorm, 0);

                    CorrelationAvg[angleIndex] += COMCorrelation(COM);
                }
            }
            // best angle for avg
            bestAngleCorAvg = FindBestAngleMax(CorrelationAvg, stepNr);
            if(params.showOutputImage)
            {
                ShowDirection(ImToShow, y, x, bestAngleCorAvg*params.angleStep, params.directionLineWidth, params.directionLineLength, params.imagesScale);
            }
            if(params.showTileOutputImage)
            {
                ShowDirectionSmall(SmallIm, bestAngleCorAvg*params.angleStep, params, minNorm, maxNorm);
                //ShowDirectionSmall(SmallIm, params.tileOffsetY, params.tileOffsetX, bestAngleCorAvg*params.angleStep, params.directionLineWidth, params.directionLineLength);
                //imshow("Small Image", ShowImageF32PseudoColor(SmallIm, minNorm, maxNorm));
            }
            if(params.showOutputImage || params.showTileOutputImage)
                waitKey(10);

            string LocalDataString;
            LocalDataString += to_string(y) + "\t";
            LocalDataString += to_string(x) + "\t";
            LocalDataString += to_string(bestAngleCorAvg) + "\t";
            LocalDataString += to_string(minNorm) + "\t";
            LocalDataString += to_string(maxNorm) + "\t";
            //LocalDataString += "n";
            OutDataString += LocalDataString;
            OutDataString += "\n";



            if(*stopCalc)
                break;

        }

        if(*stopCalc)
            break;
    }
    if(*stopCalc)
        OutDataString += "!!!!!BREAKED!!!!! \n";
    // release memory
    delete[] CorrelationAvg;
    CorrelationAvg = 0;
    ImInF.release();
    ImToShow.release();
    SmallIm.release();
    COM.release();
    SmallImToShow.release();

    if(params.textOut)
    {
        path outDir(params.OutFolderName);
        if (!exists(outDir))
        {
            OutDataString = "Error 1" + outDir.string() + " does not exists";
            return OutDataString;
        }
        if (!is_directory(outDir))
        {
            OutDataString = "Error 2" + outDir.string() + " is not a directory";
            return OutDataString;
        }

        path textOutFile = outDir;
        if(*stopCalc)
            textOutFile.append(params.FileName + "Breaked.txt");
        else
            textOutFile.append(params.FileName + ".txt");

        std::ofstream out (textOutFile.string());
        out << OutDataString;
        out.close();
    }
    return OutDataString;
}
//-----------------------------------------------------------------------------------------------


void MainWindow::ImProcess(cv::Mat ImIn,  DirDetectionParams params)
{
    ImShowPC(ImIn,params);
    ImShowGray(ImIn,params);
}
//-----------------------------------------------------------------------------------------------
void MainWindow::ReloadFileList()
{
    ui->ListWidgetFiles->clear();
    for (directory_entry& FileToProcess : directory_iterator(InputDirectory))
    {
        if (FileToProcess.path().extension() != ".tif" &&
            FileToProcess.path().extension() != ".tiff" &&
            FileToProcess.path().extension() != ".bmp" &&
            FileToProcess.path().extension() != ".jpg" &&
            FileToProcess.path().extension() != ".jpeg" &&
            FileToProcess.path().extension() != ".png" )
            continue;
        path PathLocal = FileToProcess.path();

        regex FilePattern(params.InFilePattern);
        if ((!regex_match(FileToProcess.path().filename().string().c_str(), FilePattern )))
            continue;

        if (!exists(PathLocal))
        {
            //Files << PathLocal.filename().string() << " File not exists" << "\n";
            QMessageBox msgBox;
            msgBox.setText((PathLocal.filename().string() + " File not exists" ).c_str());
            msgBox.exec();
            break;
        }
        ui->ListWidgetFiles->addItem(PathLocal.filename().string().c_str());
    }
}
//-----------------------------------------------------------------------------------------------
/*
void MainWindow::PrepareImShow()
{
    ImToShow = ShowImage16PseudoColor(ImIn, params.displayPCMin, params.displayPCMax);
    DrawTilesOnImage(ImToShow, params);
    imshow("ImOut", ImToShow);

}
*/
//-----------------------------------------------------------------------------------------------
/*
void MainWindow::ShowDirection(int y, int x, float direction, int lineWidth, int lineLength)
{
    int lineOffsetX = (int)round(lineLength * 0.5 *  sin((double)direction* PI / 180.0));
    int lineOffsetY = (int)round(lineLength * 0.5 * cos((double)direction* PI / 180.0));

    line(ImToShow, Point(x - lineOffsetX, y - lineOffsetY), Point(x + lineOffsetX, y + lineOffsetY), Scalar(0, 0.0, 0.0, 0.0), lineWidth);

    imshow("ImOut", ImToShow);
}
*/
//-----------------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------------
// system functions
//-----------------------------------------------------------------------------------------------
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{

    stopProcess = true;
    ui->setupUi(this);



    ui->comboBoxPreprocessType->addItem("Preprocess none");
    ui->comboBoxPreprocessType->addItem("Preprocess average blur");
    ui->comboBoxPreprocessType->addItem("Preprocess median blur");

    ui->comboBoxTileShape->addItem("Tile shape square");
    ui->comboBoxTileShape->addItem("Tile shape circle");
    ui->comboBoxTileShape->addItem("Tile shape hexagon");

    ui->comboBoxNormalisation->addItem("Normalization none");
    ui->comboBoxNormalisation->addItem("Normalization tile min-max");
    ui->comboBoxNormalisation->addItem("Normalization global min-max");
    ui->comboBoxNormalisation->addItem("Normalization tile +/- 3 sigma");
    ui->comboBoxNormalisation->addItem("Normalization global +/- 3 sigma");
    ui->comboBoxNormalisation->addItem("Normalization tile 1%-99%");
    ui->comboBoxNormalisation->addItem("Normalization global 1%-99%");

    ui->comboBoxScaleImages->addItem("Images scale x 4");
    ui->comboBoxScaleImages->addItem("Images scale x 2");
    ui->comboBoxScaleImages->addItem("Images scale x 1");
    ui->comboBoxScaleImages->addItem("Images scale x 1/2");
    ui->comboBoxScaleImages->addItem("Images scale x 1/4");
    ui->comboBoxScaleImages->setCurrentIndex(2);
    params.imagesScale = 1.0;

    params.DefaultParams();
    ui->comboBoxPreprocessType->setCurrentIndex(params.preprocessType);
    ui->spinBoxPreprocessKernelSize->setValue(params.preprocessKernelSize);
    ui->comboBoxTileShape->setCurrentIndex(params.tileShape);


    ui->CheckBoxShowInputImageGray->setChecked(params.showInputGray);
    ui->spinBoxMaxShowGray->setValue(params.displayGrayMax);
    ui->spinBoxMinShowGray->setValue(params.displayGrayMin);
    ui->CheckBoxShowInputImagePC->setChecked(params.showInputPC);
    ui->spinBoxMaxShowPseudoColor->setValue(params.displayPCMax);
    ui->spinBoxMinShowPseudoColor->setValue(params.displayPCMin);



    ui->spinBoxTileSize->setValue(params.tileSize);
    ui->spinBoxTileOffsetX->setValue(params.tileOffsetX);
    ui->spinBoxTileOffsetY->setValue(params.tileOffsetY);
    ui->spinBoxTileShift->setValue(params.tileShift);
    ui->CheckBoxShowTiles->setChecked(params.showTilesOnImage);
    ui->spinBoxTileLineWidth->setValue(params.tileLineWidth);
    ui->CheckBoxShowOutputImage->setChecked(params.showOutputImage);
    ui->spinBoxDirectionLineWidth->setValue(params.directionLineWidth);
    ui->spinBoxDirectionLineLenght->setValue(params.directionLineLength);
    ui->CheckBoxShowOutputTileImage->setChecked(params.showTileOutputImage);
    ui->comboBoxNormalisation->setCurrentIndex(params.normalisation);
    ui->spinBoxBinCount->setValue(params.binCount);
    ui->spinBoxMinOffset->setValue(params.minOffset);
    ui->spinBoxOffsetCount->setValue(params.offsetCount);
    ui->spinBoxOffsetStep->setValue(params.offsetStep);

    ui->CheckBoxShowOutputText->setChecked(params.showOutputText);

    InputDirectory = params.InFolderName;
    ui->LineEditInDirectory->setText(QString::fromWCharArray(InputDirectory.wstring().c_str()));
    ui->LineEditFilePattern->setText(params.InFilePattern.c_str());

    ImTemp = Mat::ones(100,100,0)*200;

    displayFlag = WINDOW_AUTOSIZE;

    stopProcess = false;
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButtonSelectInFolder_clicked()
{

    QFileDialog dialog(this, "Open Folder");
    dialog.setFileMode(QFileDialog::Directory);
    dialog.setDirectory(InputDirectory.string().c_str());

    if(dialog.exec())
    {
        InputDirectory = dialog.directory().path().toStdWString();
    }
    else
         return;

    if (!exists(InputDirectory))
    {
        QMessageBox msgBox;
        msgBox.setText((InputDirectory.string()+ " not exists ").c_str());
        msgBox.exec();
        InputDirectory = "c:\\";
    }
    if (!is_directory(InputDirectory))
    {
        QMessageBox msgBox;
        msgBox.setText((InputDirectory.string()+ " This is not a directory path ").c_str());
        msgBox.exec();
        InputDirectory = "c:\\";
    }

    ui->LineEditInDirectory->setText(QString::fromWCharArray(InputDirectory.wstring().c_str()));
    params.InFolderName = InputDirectory.string();

    ReloadFileList();
}

void MainWindow::on_ListWidgetFiles_currentTextChanged(const QString &currentText)
{
    string CurrentFileName = currentText.toStdString();
    FileToOpen = InputDirectory;
    FileToOpen.append(CurrentFileName);
    if(!exists(FileToOpen))
        return;
    ImIn = imread(FileToOpen.string().c_str(),CV_LOAD_IMAGE_ANYDEPTH);
    if(ImIn.empty())
        return;
    if(ImIn.type() != CV_16U)
    {
        ImIn.convertTo(ImIn,CV_16U);
    }

    ImProcess(ImIn,params);


}

void MainWindow::on_CheckBoxShowInputImageGray_toggled(bool checked)
{
    params.showInputGray = checked;
    if(params.showInputGray)
    {
        namedWindow("Input Gray", displayFlag);
    }
    else
    {
        destroyWindow("Input Gray");
    }
    ImProcess(ImIn,params);
}

void MainWindow::on_CheckBoxShowInputImagePC_toggled(bool checked)
{
    params.showInputPC = checked;
    if(params.showInputPC)
    {
        namedWindow("Input Pseudo Color", displayFlag);
    }
    else
    {
        destroyWindow("Input Pseudo Color");
    }
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxMinShowGray_valueChanged(int arg1)
{
    params.displayGrayMin = arg1;
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxMaxShowGray_valueChanged(int arg1)
{
    params.displayGrayMax = arg1;
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxMinShowPseudoColor_valueChanged(int arg1)
{
    params.displayPCMin = arg1;
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxMaxShowPseudoColor_valueChanged(int arg1)
{
    params.displayPCMax = arg1;
    ImProcess(ImIn,params);
}

void MainWindow::on_comboBoxPreprocessType_currentIndexChanged(int index)
{
    params.preprocessType = index;
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxPreprocessKernelSize_valueChanged(int arg1)
{
    params.preprocessKernelSize = arg1;
    ImProcess(ImIn,params);
}

void MainWindow::on_comboBoxTileShape_currentIndexChanged(int index)
{
    params.tileShape = index;
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxTileSize_valueChanged(int arg1)
{
    params.tileSize = arg1;
    if(params.tileOffsetX < params.tileSize/2)
    {
        stopProcess = true;
        params.tileOffsetX = params.tileSize/2;
        ui->spinBoxTileOffsetX->setValue(params.tileOffsetX);
        stopProcess = false;
    }
    if(params.tileOffsetY < params.tileSize/2)
    {
        stopProcess = true;
        params.tileOffsetY = params.tileSize/2;
        ui->spinBoxTileOffsetY->setValue(params.tileOffsetY);
        stopProcess = false;
    }
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxTileOffsetX_valueChanged(int arg1)
{
    if(arg1 > params.tileSize/2)
        params.tileOffsetX = arg1;
    else
    {
        params.tileOffsetX = params.tileSize/2;
        ui->spinBoxTileOffsetX->setValue(params.tileOffsetX);
    }
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxTileOffsetY_valueChanged(int arg1)
{
    if(arg1 > params.tileSize/2)
        params.tileOffsetY = arg1;
    else
    {
        params.tileOffsetY = params.tileSize/2;
        ui->spinBoxTileOffsetY->setValue(params.tileOffsetY);
    }
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxTileShift_valueChanged(int arg1)
{
    params.tileShift = arg1;
    ImProcess(ImIn,params);
}

void MainWindow::on_CheckBoxShowTiles_toggled(bool checked)
{
    params.showTilesOnImage = checked;
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxTileLineWidth_valueChanged(int arg1)
{
    params.tileLineWidth = arg1;
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxDirectionLineWidth_valueChanged(int arg1)
{
    params.directionLineWidth = arg1;
}

void MainWindow::on_spinBoxDirectionLineLenght_valueChanged(int arg1)
{
    params.directionLineLength = arg1;
}

void MainWindow::on_comboBoxNormalisation_currentIndexChanged(int index)
{
    params.normalisation = index;
}

void MainWindow::on_spinBoxBinCount_valueChanged(int arg1)
{
    params.binCount = arg1;
}

void MainWindow::on_spinBoxMinOffset_valueChanged(int arg1)
{
    params.minOffset = arg1;
}

void MainWindow::on_spinBoxOffsetCount_valueChanged(int arg1)
{
    params.offsetCount = arg1;
}

void MainWindow::on_spinBoxOffsetStep_valueChanged(int arg1)
{
    params.offsetStep = arg1;
}

void MainWindow::on_doubleSpinBoxAngleStep_valueChanged(double arg1)
{
    params.angleStep = arg1;
}

void MainWindow::on_LineEditFilePattern_returnPressed()
{
    params.InFilePattern = ui->LineEditFilePattern->text().toStdString();
    ReloadFileList();
    ImProcess(ImIn,params);
}

void MainWindow::on_pushButtonSelectOutFolder_clicked()
{
    QFileDialog dialog(this, "Open Folder");

    dialog.setFileMode(QFileDialog::Directory);
    dialog.setDirectory(OutputDirectory.string().c_str());

    if(dialog.exec())
    {
        OutputDirectory = dialog.directory().path().toStdWString();
    }
    else
         return;

    //InputDirectory = dialog.getExistingDirectory().toStdWString();//  toStdString());
    if (!exists(OutputDirectory))
    {
        QMessageBox msgBox;
        msgBox.setText((OutputDirectory.string()+ " not exists ").c_str());
        msgBox.exec();
        OutputDirectory = "c:\\";
    }
    if (!is_directory(OutputDirectory))
    {
        QMessageBox msgBox;
        msgBox.setText((OutputDirectory.string()+ " This is not a directory path ").c_str());
        msgBox.exec();
        OutputDirectory = "c:\\";
    }
    ui->LineEditOutDirectory->setText(QString::fromWCharArray(OutputDirectory.wstring().c_str()));
    params.OutFolderName = OutputDirectory.string();
}

void MainWindow::on_CheckBoxShowOutputText_toggled(bool checked)
{
    params.showOutputText = checked;
}

void MainWindow::on_pushButtonCalculateDorectionality_clicked()
{
    breakProcess = false;
    time_t begin,end;
    time (&begin);
    params.FileName = FileToOpen.filename().string();

    string OutStr = DirEstimation(ImIn, params, &breakProcess);
    time (&end);
    double difference = difftime (end,begin);
    string timeString = "calcTime = " + to_string(difference) + " s";
    ui->textEditOutput->setText(timeString.c_str());
    ui->textEditOutput->append(OutStr.c_str());
}

void MainWindow::on_pushButtonCalculateDirectionalityForAll_clicked()
{
    breakProcess = false;
    int filesCount = ui->ListWidgetFiles->count();
    int firstFile = ui->spinBoxFirstFileToProcess->value();
    ui->textEditOutput->clear();
    for(int fileNr = firstFile; fileNr< filesCount; fileNr++)
    {
        time_t begin,end;
        time (&begin);
        path LocalFileToOpen = params.InFolderName;
        params.FileName = ui->ListWidgetFiles->item(fileNr)->text().toStdString();
        LocalFileToOpen.append(params.FileName);
        Mat LocalIm = imread(LocalFileToOpen.string().c_str(),CV_LOAD_IMAGE_ANYDEPTH);

        if(LocalIm.type() != CV_16U)
        {
            LocalIm.convertTo(LocalIm,CV_16U);
        }


        DirEstimation(LocalIm, params, &breakProcess);
        time (&end);
        double difference = difftime (end,begin);
        string timeString = params.FileName + " calcTime = " + to_string(difference) + " s" + "\n";
        ui->textEditOutput->append(timeString.c_str());
        ui->lineEditCurrentFile->setText(params.FileName.c_str());
        waitKey(20);
        if( breakProcess)
            break;
    }
}

void MainWindow::on_CheckBoxShowOutputImage_toggled(bool checked)
{
    params.showOutputImage = checked;
    if(params.showOutputImage)
    {
        namedWindow("ImOut", displayFlag);
    }
    else
    {
        destroyWindow("ImOut");
    }
}

void MainWindow::on_CheckBoxShowOutputTileImage_toggled(bool checked)
{
    params.showTileOutputImage = checked;
    if(params.showTileOutputImage)
    {
        namedWindow("ImSmall", displayFlag);
    }
    else
    {
        destroyWindow("ImSmall");
    }
}


void MainWindow::on_pushButtonSop_clicked()
{
    breakProcess = true;
}

void MainWindow::on_comboBoxScaleImages_currentIndexChanged(int index)
{
    switch(index)
    {
    case 0:
        params.imagesScale = 4.0;
        break;
    case 1:
        params.imagesScale = 2.0;
        break;
    case 2:
        params.imagesScale = 1.0;
        break;

    case 3:
        params.imagesScale = 1.0/2.0;
        break;
    case 4:
        params.imagesScale = 1.0/4.0;
        break;
    default:
        params.imagesScale = 1.0;
        break;
    }
    ImShowPC(ImIn,params);
    ImShowGray(ImIn,params);
}
