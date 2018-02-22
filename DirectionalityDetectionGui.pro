#-------------------------------------------------
#
# Project created by QtCreator 2018-02-16T20:48:08
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = DirectionalityDetectionGui
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp\
        ../../ProjectsLib/LibMarcin/DispLib.cpp \
        ../../ProjectsLib/LibMarcin/RegionU16Lib.cpp \
        ../../ProjectsLib/LibMarcin/NormalizationLib.cpp \
        ../../ProjectsLib/LibMarcin/HaralickLib.cpp \
        dirdetectionparams.cpp

HEADERS  += mainwindow.h\
        ../../ProjectsLib/LibMarcin/DispLib.h \
        ../../ProjectsLib/LibMarcin/RegionU16Lib.h \
        ../../ProjectsLib/LibMarcin/NormalizationLib.h \
        ../../ProjectsLib/LibMarcin/HaralickLib.h \
        dirdetectionparams.h

FORMS    += mainwindow.ui

win32: INCLUDEPATH += C:\opencv\build\include\
win32: INCLUDEPATH += C:\Boost\include\boost-1_62\
win32: INCLUDEPATH += C:\LibTiff\Include\
win32: INCLUDEPATH += ../../ProjectsLib\LibMarcin\
win32: INCLUDEPATH += ../../ProjectsLibForein/LibPMS/

win32: LIBS += -LC:/opencv/build/x64/vc12/lib
win32: LIBS += -lopencv_core2413d
win32: LIBS += -lopencv_highgui2413d
win32: LIBS += -lopencv_imgproc2413d

win32: LIBS += -LC:\Boost\lib
win32:  LIBS += -lboost_filesystem-vc120-mt-gd-1_62
win32:  LIBS += -lboost_regex-vc120-mt-gd-1_62
