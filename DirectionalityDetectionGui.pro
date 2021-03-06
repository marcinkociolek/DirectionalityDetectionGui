#-------------------------------------------------
#
# Project created by QtCreator 2018-02-16T20:48:08
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = DirectionalityDetectionGui
TEMPLATE = app

DEFINES += QT_DEPRECATED_WARNINGS

SOURCES += main.cpp\
        mainwindow.cpp\
        ../../ProjectsLib/LibMarcin/DispLib.cpp \
        ../../ProjectsLib/LibMarcin/RegionU16Lib.cpp \
        ../../ProjectsLib/LibMarcin/NormalizationLib.cpp \
        ../../ProjectsLib/LibMarcin/HaralickLib.cpp \
        ../../ProjectsLib/LibMarcin/dirdetectionparams.cpp \
        ../../ProjectsLib/LibMarcin/displayfordirdetection.cpp

HEADERS  += mainwindow.h\
        ../../ProjectsLib/LibMarcin/DispLib.h \
        ../../ProjectsLib/LibMarcin/RegionU16Lib.h \
        ../../ProjectsLib/LibMarcin/NormalizationLib.h \
        ../../ProjectsLib/LibMarcin/HaralickLib.h \
        ../../ProjectsLib/LibMarcin/dirdetectionparams.h \
    ../../ProjectsLib/LibMarcin/displayfordirdetection.h

FORMS    += mainwindow.ui


INCLUDEPATH += /usr/include/boost/
INCLUDEPATH += ../../ProjectsLib/LibMarcin/
INCLUDEPATH += ../SkopiaSegment/
INCLUDEPATH +=/home/marcin/Documents/ProjectsForeign/usr/opencv541CudaDynamicGTK2/include/opencv4/


LIBS += -ltiff
LIBS += -lboost_filesystem
LIBS += -lboost_regex

LIBS += -L/home/marcin/Documents/ProjectsForeign/usr/opencv541CudaDynamicGTK2/lib/

LIBS += -lopencv_highgui
LIBS += -lopencv_core
LIBS += -lopencv_imgproc

LIBS += -lopencv_features2d
LIBS += -lopencv_imgcodecs

LIBS += -lopencv_dnn

qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
