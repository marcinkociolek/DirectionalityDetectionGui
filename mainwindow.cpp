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


using namespace std;
using namespace boost::filesystem;
using namespace boost;
using namespace cv;
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
                int edgeLength = params.tileWidth;
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
void MainWindow::ImProcess(cv::Mat ImIn,  DirDetectionParams params)
{
    if(stopProcess)
        return;
    ImShowPC(ImIn,params);

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

void MainWindow::on_doubleSpinBox_valueChanged(double arg1)
{
    params.angleStep = arg1;
    ImProcess(ImIn,params);
}
