#include "mainwindow.h"
#include "ui_mainwindow.h"


#include <QFileDialog>
#include <QMessageBox>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "DispLib.h"
#include "NormalizationLib.h"

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
    case 1: // Rectangle
        Roi = Mat::ones(size, size, CV_8U);
        break;
    case 2: // Ellipse
        Roi = Mat::zeros(size, size, CV_8U);
        ellipse(Roi, Point(size / 2, size / 2),	Size(size / 2, size / 2), 0.0, 0.0, 360.0, 1, -1);
        break;
    case 3: // Hexagon
    {
        int edgeLength = size/2;
        int roiMaxX = size;
        int roiMaxY = (int)((double)size * 0.8660254);
        Roi = Mat::zeros(roiMaxY, roiMaxX, CV_8U);

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
        unsigned char *wRoi;

        for (int y = 1; y < roiMaxY - 1; y++)
        {
            wRoi = (unsigned char *)Roi.data + roiMaxX * y;
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
    default:
        Roi = Mat();
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
    case 1:
        NormParamsMinMax(ImF, maxNormGlobal, minNormGlobal);
        break;
    case 3:
        NormParamsMeanP3Std(ImF, maxNormGlobal, minNormGlobal);
        break;
    case 5:
        NormParams1to99perc(ImF, maxNormGlobal, minNormGlobal);
        break;
    default:
        break;
    }
}

//-----------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------
// my functions
//-----------------------------------------------------------------------------------------------
void DrawTilesOnImage(cv::Mat ImIn, DirDetectionParams params)
{
    if(!params.showTiles)
        return;
    if(ImIn.empty())
        return;
    int maxX = ImIn.cols;
    int maxY = ImIn.rows;

    if(!maxY || !maxY)
        return;
    switch (params.tileShape)
    {
    case 1:
        for (int y = params.tileOffsetY; y <= (maxY - params.tileOffsetY); y += params.tileShiftY)
        {
            for (int x = params.tileOffsetX; x <= (maxX - params.tileOffsetX); x += params.tileShiftX)
            {
                rectangle(ImIn, Point(x - params.tileWidth / 2, y - params.tileHeight / 2),
                    Point(x - params.tileWidth / 2 + params.tileWidth - 1, y - params.tileHeight / 2 + params.tileHeight - 1),
                    Scalar(0.0, 0.0, 0.0, 0.0), params.tileLineWidth);
            }
        }
        break;
    case 2:
        for (int y = params.tileOffsetY; y <= (maxY - params.tileOffsetY); y += params. tileShiftY)
        {
            for (int x = params.tileOffsetX; x <= (maxX - params.tileOffsetX); x += params.tileShiftX)
            {
                ellipse(ImIn, Point(x, y),
                    Size(params.tileWidth / 2, params.tileHeight / 2), 0.0, 0.0, 360.0,
                    Scalar(0.0, 0.0, 0.0, 0.0), params.tileLineWidth);
            }
        }
        break;
    case 3:
        for (int y = params.tileOffsetY; y <= (maxY - params.tileOffsetY); y += params.tileShiftY)
        {
            for (int x = params.tileOffsetX; x <= (maxX - params.tileOffsetX); x += params.tileShiftX)
            {
                int edgeLength = params.tileWidth/2;
                Point vertice0(x - edgeLength / 2, y - (int)((float)edgeLength * 0.8660254));
                Point vertice1(x + edgeLength - edgeLength / 2, y - (int)((float)edgeLength * 0.8660254));
                Point vertice2(x + edgeLength, y);
                Point vertice3(x + edgeLength - edgeLength / 2, y + (int)((float)edgeLength * 0.8660254));
                Point vertice4(x - edgeLength / 2, y + (int)((float)edgeLength * 0.8660254));
                Point vertice5(x - edgeLength, y);

                line(ImIn, vertice0, vertice1, Scalar(0.0, 0.0, 0.0, 0.0), params.tileLineWidth);
                line(ImIn, vertice1, vertice2, Scalar(0.0, 0.0, 0.0, 0.0), params.tileLineWidth);
                line(ImIn, vertice2, vertice3, Scalar(0.0, 0.0, 0.0, 0.0), params.tileLineWidth);
                line(ImIn, vertice3, vertice4, Scalar(0.0, 0.0, 0.0, 0.0), params.tileLineWidth);
                line(ImIn, vertice4, vertice5, Scalar(0.0, 0.0, 0.0, 0.0), params.tileLineWidth);
                line(ImIn, vertice5, vertice0, Scalar(0.0, 0.0, 0.0, 0.0), params.tileLineWidth);
            }
        }
        break;
    default:
        break;
    }

}

//-----------------------------------------------------------------------------------------------
void ImShowPC(cv::Mat ImIn,  DirDetectionParams params)
{
    if(!params.showInputPC)
        return;
    if(ImIn.empty())
        return;
    Mat ImToShow = ShowImage16PseudoColor(ImIn, params.displayPCMin, params.displayPCMax);
    DrawTilesOnImage(ImToShow, params);
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
    Mat ImToShow = ShowImage16Gray(ImIn, params.displayPCMin, params.displayPCMax);
    DrawTilesOnImage(ImToShow, params);
    imshow("Input Gray", ImToShow);
    ImToShow.release();
}
//-----------------------------------------------------------------------------------------------
void MainWindow::ImProcess(cv::Mat ImIn,  DirDetectionParams params)
{
    if(stopProcess)
        return;
    ImShowPC(ImIn,params);
    ImShowGray(ImIn,params);

    Roi = CreateStandardROI(params.tileWidth, params.tileShape);
    imshow("Roi", Roi*255);

    int stepNr = (int)(180.0 / params.angleStep); // angle step for computations (number of steps)

    float *CorrelationAvg = new float[stepNr];
    int *AnglesAvg = new int[stepNr]; // vector for best angles histogtam
    //Matrix declarations
    Mat ImInF, ImToShow, SmallIm, COM, SmallImToShow;

    ImIn.convertTo(ImInF, CV_32F);


    float maxNormGlobal = 65535;
    float minNormGlobal = 0;

    GlobalNormalisation(ImInF, params.normalisation , &maxNormGlobal, &minNormGlobal);

    ParamsString.empty();
    ParamsString = params.ShowParams();

    string OutDataString = ParamsString;
    OutDataString += "FileName \t" + FileToOpen.string();
    OutDataString += "Tile Y\tTile X\t";
    OutDataString += "Angle \t";
    OutDataString += "Mean Intensity\tTile min norm\tTile max norm\t";
    OutDataString += "\n";

    for (int y = params.tileOffsetX; y <= (maxY - params.tileOffsetX); y += params.tileShiftX)
    {
        for (int x = params.tileOffsetX; x <= (maxX - params.tileOffsetX); x += params.tileShiftX)
        {
            ImInF(Rect(x - Roi.cols / 2, y - Roi.rows / 2, Roi.cols, Roi.rows)).copyTo(SmallIm);
            float maxNorm, minNorm;
            switch (ProcOptions.normalisation)
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

                // ofset loop
            for (int offset = ProcOptions.minOfset; offset <= ProcOptions.maxOfset; offset += 1)
            {
                for (int angleIndex = 0; angleIndex < stepNr; angleIndex++)
                {
                    float angle = ProcOptions.angleStep * angleIndex;

                    COM.release();

                    if (ProcOptions.tileShape < 2)
                        COM = COMCardone4(SmallIm, offset, angle, ProcOptions.binCount, maxNorm, minNorm);
                    else
                        COM = COMCardoneRoi(SmallIm, Roi, offset, angle, ProcOptions.binCount, maxNorm, minNorm, 1);

                    CorrelationAvg[angleIndex] = COMCorrelation(COM);
                }
            }
            // best angle for avg
            bestAngleCorAvg = FindBestAngleMax(CorrelationAvg, stepNr);

    }
    // release memory
    delete[] CorrelationAvg;
    delete[] AnglesAvg;
    ImInF.release();
    ImToShow.release();
    SmallIm.release();
    COM.release();
    SmallImToShow.release();
}
//-----------------------------------------------------------------------------------------------
void MainWindow::ReloadFileList()
{
    ui->ListWidgetFiles->clear();
    for (directory_entry& FileToProcess : directory_iterator(InputDirectory))
    {
        if (FileToProcess.path().extension() != ".tif" && FileToProcess.path().extension() != ".png" )
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

    ui->comboBoxNormalisation->addItem("Normalisation none");
    ui->comboBoxNormalisation->addItem("Normalisation tile min-max");
    ui->comboBoxNormalisation->addItem("Normalisation global min-max");
    ui->comboBoxNormalisation->addItem("Normalisation tile +/- 3 sigma");
    ui->comboBoxNormalisation->addItem("Normalisation global +/- 3 sigma");

    params.DefaultParams();

    ui->CheckBoxShowInputImageGray->setChecked(params.showInputGray);
    ui->CheckBoxShowInputImagePC->setChecked(params.showInputPC);
    ui->spinBoxMaxShowGray->setValue(params.displayGrayMax);
    ui->spinBoxMinShowGray->setValue(params.displayGrayMin);
    ui->spinBoxMaxShowPseudoColor->setValue(params.displayPCMax);
    ui->spinBoxMinShowPseudoColor->setValue(params.displayPCMin);
    ui->comboBoxPreprocessType->setCurrentIndex(params.preprocessType);
    ui->spinBoxPreprocessKernelSize->setValue(params.preprocessKernelSize);
    ui->comboBoxTileShape->setCurrentIndex(params.tileShape - 1);
    ui->spinBoxTileWidth->setValue(params.tileWidth);
    ui->spinBoxTileHeight->setValue(params.tileHeight);
    ui->spinBoxTileOffsetX->setValue(params.tileOffsetX);
    ui->spinBoxTileOffsetY->setValue(params.tileOffsetY);
    ui->spinBoxTileShiftX->setValue(params.tileShiftX);
    ui->spinBoxTileShiftY->setValue(params.tileShiftY);
    ui->CheckBoxShowTiles->setChecked(params.showTiles);
    ui->spinBoxTileLineWidth->setValue(params.tileLineWidth);
    ui->CheckBoxShowDirection->setChecked(params.showDirection);
    ui->spinBoxDirectionLineWidth->setValue(params.directionLineWidth);
    ui->spinBoxDirectionLineLenght->setValue(params.directionLineLength);
    ui->comboBoxNormalisation->setCurrentIndex(params.normalisation);
    ui->spinBoxBinCount->setValue(params.binCount);
    ui->spinBoxMinOffset->setValue(params.minOffset);
    ui->spinBoxOffsetCount->setValue(params.offsetCount);
    ui->spinBoxOffsetStep->setValue(params.offsetStep);

    InputDirectory = params.InFolderName;
    ui->LineEditInDirectory->setText(QString::fromWCharArray(InputDirectory.wstring().c_str()));
    ui->LineEditFilePattern->setText(params.InFilePattern.c_str());
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

    //InputDirectory = dialog.getExistingDirectory().toStdWString();//  toStdString());
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
    ImProcess(ImIn,params);


}

void MainWindow::on_CheckBoxShowInputImageGray_toggled(bool checked)
{
    params.showInputGray = checked;
    ImProcess(ImIn,params);
}

void MainWindow::on_CheckBoxShowInputImagePC_toggled(bool checked)
{
    params.showInputPC = checked;
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
    params.tileShape = index + 1;
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxTileWidth_valueChanged(int arg1)
{
    params.tileWidth = arg1;
    if(params.tileOffsetX < params.tileWidth/2)
    {
        stopProcess = true;
        params.tileOffsetX = params.tileWidth/2;
        ui->spinBoxTileOffsetX->setValue(params.tileOffsetX);
        stopProcess = false;
    }
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxTileHeight_valueChanged(int arg1)
{
    params.tileHeight = arg1;
    if(params.tileOffsetY < params.tileHeight/2)
    {
        stopProcess = true;
        params.tileOffsetY = params.tileHeight/2;
        ui->spinBoxTileOffsetY->setValue(params.tileOffsetY);
        stopProcess = false;
    }
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxTileOffsetX_valueChanged(int arg1)
{
    if(arg1 > params.tileWidth/2)
        params.tileOffsetX = arg1;
    else
    {
        params.tileOffsetX = params.tileWidth/2;
        ui->spinBoxTileOffsetX->setValue(params.tileOffsetX);
    }
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxTileOffsetY_valueChanged(int arg1)
{
    if(arg1 > params.tileHeight/2)
        params.tileOffsetY = arg1;
    else
    {
        params.tileOffsetY = params.tileHeight/2;
        ui->spinBoxTileOffsetY->setValue(params.tileOffsetY);
    }
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxTileShiftX_valueChanged(int arg1)
{
    params.tileShiftX = arg1;
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxTileShiftY_valueChanged(int arg1)
{
    params.tileShiftY = arg1;
    ImProcess(ImIn,params);
}

void MainWindow::on_CheckBoxShowTiles_toggled(bool checked)
{
    params.showTiles = checked;
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxTileLineWidth_valueChanged(int arg1)
{
    params.tileLineWidth = arg1;
    ImProcess(ImIn,params);
}

void MainWindow::on_CheckBoxShowDirection_toggled(bool checked)
{
    params.showDirection = checked;
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxDirectionLineWidth_valueChanged(int arg1)
{
    params.directionLineWidth = arg1;
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxDirectionLineLenght_valueChanged(int arg1)
{
    params.directionLineLength = arg1;
    ImProcess(ImIn,params);
}

void MainWindow::on_comboBoxNormalisation_currentIndexChanged(int index)
{
    params.normalisation = index;
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxBinCount_valueChanged(int arg1)
{
    params.binCount = arg1;
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxMinOffset_valueChanged(int arg1)
{
    params.minOffset = arg1;
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxOffsetCount_valueChanged(int arg1)
{
    params.offsetCount = arg1;
    ImProcess(ImIn,params);
}

void MainWindow::on_spinBoxOffsetStep_valueChanged(int arg1)
{
    params.offsetStep = arg1;
    ImProcess(ImIn,params);
}

void MainWindow::on_doubleSpinBoxAngleStep_valueChanged(double arg1)
{
    params.angleStep = arg1;
    ImProcess(ImIn,params);
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
    //dialog.setDirectory(InputDirectory.string().c_str());

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
    params.OutFolderName1 = OutputDirectory.string();
}
