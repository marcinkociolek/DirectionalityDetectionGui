#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>

#include "dirdetectionparams.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    DirDetectionParams params;

    bool stopProcess;

    cv::Mat ImIn;
    //std::string CurrentFileName;
    boost::filesystem::path FileToOpen;
    boost::filesystem::path CurrentDir;
    boost::filesystem::path InputDirectory;

    void ImProcess(cv::Mat ImIn,  DirDetectionParams params);

    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_pushButtonSelectInFolder_clicked();

    void on_ListWidgetFiles_currentTextChanged(const QString &currentText);

    void on_CheckBoxShowInputImageGray_toggled(bool checked);

    void on_CheckBoxShowInputImagePC_toggled(bool checked);

    void on_spinBoxMinShowGray_valueChanged(int arg1);

    void on_spinBoxMaxShowGray_valueChanged(int arg1);

    void on_spinBoxMinShowPseudoColor_valueChanged(int arg1);

    void on_spinBoxMaxShowPseudoColor_valueChanged(int arg1);

    void on_comboBoxPreprocessType_currentIndexChanged(int index);

    void on_spinBoxPreprocessKernelSize_valueChanged(int arg1);

    void on_comboBoxTileShape_currentIndexChanged(int index);

    void on_spinBoxTileWidth_valueChanged(int arg1);

    void on_spinBoxTileHeight_valueChanged(int arg1);

    void on_spinBoxTileOffsetX_valueChanged(int arg1);

    void on_spinBoxTileOffsetY_valueChanged(int arg1);

    void on_spinBoxTileShiftX_valueChanged(int arg1);

    void on_spinBoxTileShiftY_valueChanged(int arg1);

    void on_CheckBoxShowTiles_toggled(bool checked);

    void on_spinBoxTileLineWidth_valueChanged(int arg1);

    void on_CheckBoxShowDirection_toggled(bool checked);

    void on_spinBoxDirectionLineWidth_valueChanged(int arg1);

    void on_spinBoxDirectionLineLenght_valueChanged(int arg1);

    void on_comboBoxNormalisation_currentIndexChanged(int index);

    void on_spinBoxBinCount_valueChanged(int arg1);

    void on_spinBoxMinOffset_valueChanged(int arg1);

    void on_spinBoxOffsetCount_valueChanged(int arg1);

    void on_spinBoxOffsetStep_valueChanged(int arg1);

    void on_doubleSpinBox_valueChanged(double arg1);

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H