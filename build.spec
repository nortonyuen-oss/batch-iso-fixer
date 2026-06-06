# -*- mode: python ; coding: utf-8 -*-

import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


block_cipher = None
datas = [
    ("app.py", "."),
    ("image_processor.py", "."),
]
binaries = []
hiddenimports = []

for package in ("streamlit", "altair"):
    datas += collect_data_files(package)
    hiddenimports += collect_submodules(package)

for package in ("streamlit", "altair", "pyarrow", "pandas", "Pillow", "opencv-python", "numpy"):
    try:
        datas += copy_metadata(package)
    except Exception:
        pass


a = Analysis(
    ["launcher.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BatchIsoFixer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="BatchIsoFixer",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="BatchIsoFixer.app",
        icon=None,
        bundle_identifier="com.nortonyuen.batchisofixer",
    )
