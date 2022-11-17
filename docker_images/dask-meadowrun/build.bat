REM this has to be run from the root of the repo

set /p MEADOWRUN_VERSION_FILE=<src/meadowrun/version.py
set MEADOWRUN_VERSION=%MEADOWRUN_VERSION_FILE:~15,-1%

echo Meadowrun version: %MEADOWRUN_VERSION%

docker build -t meadowrun/dask-meadowrun:%MEADOWRUN_VERSION%-py3.9-3 --build-arg PYTHON_VERSION=3.9 --build-arg MEADOWRUN_VERSION=%MEADOWRUN_VERSION% -f docker_images/dask-meadowrun/DaskMeadowrunDockerfile docker_images/dask-meadowrun
docker push meadowrun/dask-meadowrun:%MEADOWRUN_VERSION%-py3.9-3
