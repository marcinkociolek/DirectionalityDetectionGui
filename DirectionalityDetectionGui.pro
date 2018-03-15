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
        dirdetectionparams.cpp

HEADERS  += mainwindow.h\
        ../../ProjectsLib/LibMarcin/DispLib.h \
        ../../ProjectsLib/LibMarcin/RegionU16Lib.h \
        ../../ProjectsLib/LibMarcin/NormalizationLib.h \
        ../../ProjectsLib/LibMarcin/HaralickLib.h \
        dirdetectionparams.h

FORMS    += mainwindow.ui


win32: INCLUDEPATH += C:\opencv\build\include\
win32: INCLUDEPATH += C:\boost_1_66_0\
win32: INCLUDEPATH += C:\LibTiff\Include\
win32: INCLUDEPATH += ../../ProjectsLib\LibMarcin\
win32: INCLUDEPATH += ../../ProjectsLibForein/LibPMS/

# this is for debug
#win32: LIBS += -LC:/opencv/build/x64/vc15/lib/
#win32: LIBS += -lopencv_world341d

#win32: LIBS += -LC:/boost_1_66_0/stage/x64/lib/
#win32:  LIBS += -lboost_filesystem-vc141-mt-gd-x64-1_66
#win32:  LIBS += -lboost_regex-vc141-mt-gd-x64-1_66


# this is for release
win32: LIBS += -LC:/opencv/build/x64/vc15/lib/
win32: LIBS += -lopencv_world341

win32: LIBS += -LC:/boost_1_66_0/stage/x64/lib/
win32:  LIBS += -lboost_filesystem-vc141-mt-x64-1_66
win32:  LIBS += -lboost_regex-vc141-mt-x64-1_66

