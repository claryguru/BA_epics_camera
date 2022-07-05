@echo off
@REM Phoebus launcher for Windows
@REM Uses a JDK that's located next to this folder,
@REM otherwise assumes JDK is on the PATH

REM remember script's path, because shift will change %0
set scriptPath=%~dp0

%~d0
cd %~dp0

SETLOCAL EnableDelayedExpansion

IF EXIST %scriptPath%\..\jdk (
    set JAVA_HOME=%scriptPath%\..\jdk
    set "PATH=!JAVA_HOME!\bin;%PATH%"
    @echo Found JDK !JAVA_HOME!
)

if EXIST "update" (
    ECHO Installing update...
    rd /S/Q doc
    rd /S/Q lib
    del *.jar
    move /Y update\*.* .
    move /Y update\doc .
    move /Y update\lib .
    move /Y update\config-files\* .\config-files
    rmdir update\config-files
    rmdir update
    ECHO Updated.
)

java -version

if ERRORLEVEL 1 (
	echo Java not found
	set /p DUMMY=Hit ENTER to close...
	EXIT /B
)

FOR %%F in (*.jar) DO (SET JAR=%%F)

REM Don't start CA repeater (#494)
REM To get one instance, use server mode
REM and pass all arguments along

echo Starting phoebus using %JAR% with
echo %*

java -DCA_DISABLE_REPEATER=true -jar %JAR% %*

REM alternative invocation, which uses javafx modules from javafx skd (might be useful to quelch warnings or if distributed with JRE) 
REM java --module-path %scriptPath%\..\javafx-sdk\lib --add-modules javafx.controls,javafx.fxml,javafx.web -DCA_DISABLE_REPEATER=true -jar %JAR% -server 4918 %args%
@echo on
