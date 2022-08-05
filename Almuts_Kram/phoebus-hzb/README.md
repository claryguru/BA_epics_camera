# HZB-Phoebus

This is a build of phoebus with a custom launcher, loading custom default
settings for use at HZB facilities MLS, BESSY, BerlinPro/SeaLab and Hobicat.

## Requirements

A java runtime or jdk with versions >= 11 on your path.
For Windows, Oracle does not provide a JRE anymore, so please install the JDK. 
On Windows you may place the JDK in a directory jdk, next to this one. Like so

C:\a\path\to\somewhere
├── phoebus-hzb
│   └── phoebus.bat
└── jdk

The startup script will then find the java executable there.

Alternatively, and if on Linux or MacOS, ensure that java is on your path.

## Usage

Depending on your platform, run the phoebus.sh or phoebus.bat scripts.
phoebus_w.bat is an alternative which hides the console windows.
These scripts simply ensure java is present and then launch Phoebus.

See the next section on the specifics of the HZB-Launcher or run with 
`--help` or `-h` to see all options.

## HZB Launcher

The Phoebus launcher here provides additional options and functionality over
the default launcher. All options and arguments that work for the default
Phoebus should work here as well.

It loads HZB-specific default settings depending on the system
you wish to access, like, e.g., `bessy` or `mls`, and the access mode: external, 
internal or direct.
The system and access mode are determined in order from the `--access` and 
`--system` options, the environment or inferred from the host- and username.

Addtional settings files provided on the command line are loaded after these
defaults and may thus override any of the default settings.

Furthermore, this launcher interprets all non-option command line arguments as
ressources. Thus you can run
```
./phoebus.sh -ce ~/work/displays/*.bob
```

To open all .bob files in `~/work/displays` for editing (`-e`) while not restoring
the previous session (`-c`).
