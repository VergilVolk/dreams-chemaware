import platform
import pathlib

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
