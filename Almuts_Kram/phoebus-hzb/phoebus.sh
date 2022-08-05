#!/bin/sh
#
# Phoebus launcher for Linux or Mac OS X

# Set PHOEBUS_HOME on the installation system if you need to use symlinks
# otherwise auto find of the script's path should work
if [ -d "$PHOEBUS_HOME" ]
then
  TOP="$PHOEBUS_HOME"
else
  TOP="$( cd "$(dirname "$0")" ; pwd -P )"
fi

if [ -d "${TOP}/target" ]
then
  TOP="$TOP/target"
fi

if [ -d "${TOP}/update" ]
then
  echo "Installing update..."
  rm -rf ${TOP}/doc ${TOP}/lib ${TOP}/config-files
  rm *.jar
  mv ${TOP}/update/* ${TOP}/
  rmdir ${TOP}/update
  echo "Updated."
fi

JAR=`ls ${TOP}/*.jar | sort | tail -1`

# assert that java is found
if ! java -version >/dev/null 2>&1
then
  echo "java not on path, can't run phoebus"
  exit 1
fi

# Old way
# check arguments
# runBackground="true"
# ARGS=$@
# if test "${ARGS#*-h}" != "$ARGS" -o "${ARGS#*-main}" != "$ARGS" -o "${ARGS#*-V}" != "$ARGS" -o "${ARGS#*--version}" != "$ARGS"; then
#   runBackground="false"
# fi

# if [ $runBackground = "true" ]
# then
#   java -jar $JAR "$@" &
# else
#   java -jar $JAR "$@"
# fi

# Just run it in foreground, if someone wants things in background, they can start the script in the background
java -jar $JAR "$@"
