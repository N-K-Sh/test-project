#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
one_way_sync.py v1.4.1 — односторонняя синхронизация: флеш-диск (источник) -> папка (приёмник)

Ключевые возможности:
- Источник «главный»: приёмник приводится в точное соответствие источнику (создание/замена/удаление, рекурсивно).
- Сравнение изменений: time-field (mtime|ctime) + epsilon, опц. размер, опц. хэш (never|if-needed|always|verify-after-copy).
- Интерактивный выбор путей (Windows: список дисков), dry-run, подтверждение перед удалениями, лог-файл.
- Прогресс-бар (--progress), гарантированная пауза в конце (по умолчанию; отключить --no-pause).
- Windows-фиксы: длинные пути (\\?\\ и \\?\\UNC\\...), регистронезависимые сравнения, снятие read-only, пост-проверка «хвостов».

Python ≥ 3.9, только стандартная библиотека. Поддержка Windows/Linux/macOS.
"""

import argparse, hashlib, os, shutil, sys, time, stat
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple, Optional, Dict

VERSION = "1.4.1"

# ------------------------- ОС/время/хэши -------------------------

def is_windows() -> bool:
    return os.name == "nt"

def human_ts(ts: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    except Exception:
        return str(ts)

def _win_long(p: Path) -> str:
    r"""Вернуть путь в формате \\?\\... (или \\?\\UNC\\server\\share\\...) на Windows; на других ОС — обычный путь."""
    s = os.fspath(p)
    if not is_windows():
        return s
    s = os.path.abspath(s)
    if s.startswith("\\\\?\\"):
        return s
    if s.startswith("\\\\"):  # UNC
        return "\\\\?\\UNC\\" + s[2:]
    return "\\\\?\\" + s

def _makedirs(path: Path) -> None:
    """Создать каталоги (с длинными путями на Windows)."""
    if is_windows():
        os.makedirs(_win_long(path), exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)

def file_time(path: Path, mode: str) -> float:
    """Возвращает временную метку; на Windows использует длинные пути."""
    if is_windows():
        st = os.stat(_win_long(path))
    else:
        st = path.stat()
    if mode == "mtime":
        return st.st_mtime
    if is_windows():
        return st.st_ctime  # creation time на Windows
    bt = getattr(st, "st_birthtime", None)
    return bt if bt is not None else st.st_mtime

def hash_file(path: Path, algo: str = "blake2b", chunk_mb: int = 8) -> str:
    h = hashlib.blake2b() if algo == "blake2b" else hashlib.sha256()
    chunk = 1024 * 1024 * max(1, int(chunk_mb))
    if is_windows():
        fpath = _win_long(path)
        f = open(fpath, "rb")
    else:
        f = path.open("rb")
    with f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

# ------------------------- Прогресс-бар -------------------------

class Progress:
    def __init__(self, total: int, enabled: bool):
        self.total = max(0, int(total))
        self.enabled = enabled and self.total > 0
        self.current = 0
    def step(self, msg: Optional[str] = None) -> None:
        if not self.enabled: return
        self.current += 1
        if self.current > self.total: self.total = self.current
        pct = int(self.current * 100 / self.total) if self.total else 100
        line = f"\rПрогресс: {self.current}/{self.total} ({pct}%)"
        if msg:
            if len(msg) > 60: msg = msg[:57] + "..."
            line += f" — {msg}"
        print(line, end="", flush=True)
    def close(self):
        if self.enabled: print()

# ------------------------- Вспомогательное -------------------------

def norm_key(rel: Path) -> str:
    """Ключ сравнения: на Windows без учёта регистра."""
    s = rel.as_posix()
    return s.lower() if is_windows() else s

def index_by_key(rel_paths: Set[Path]) -> Dict[str, Path]:
    return {norm_key(p): p for p in rel_paths}

def under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve()); return True
    except Exception:
        return False

# ------------------------- Сканирование деревьев -------------------------

def collect_tree(root: Path) -> Tuple[Set[Path], Set[Path]]:
    files: Set[Path] = set(); dirs: Set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        for d in dirnames:
            p = dp / d
            if p.is_symlink(): continue
            dirs.add(p.relative_to(root))
        for f in filenames:
            p = dp / f
            if p.is_symlink(): continue
            files.add(p.relative_to(root))
    return files, dirs

# ------------------------- Валидация и лог -------------------------

def validate_paths(src: Path, dst: Path) -> None:
    if not src.exists() or not src.is_dir():
        sys.exit(f"Ошибка: источник не существует или не папка: {src}")
    if src.resolve() == dst.resolve():
        sys.exit("Ошибка: источник и приёмник совпадают.")
    if under(dst, src):
        sys.exit("Ошибка: приёмник расположен внутри источника — так нельзя.")
    if under(src, dst):
        sys.exit("Ошибка: источник расположен внутри приёмника — так нельзя.")

def log_append(log: List[str], line: str, to_console: bool = True) -> None:
    log.append(line)
    if to_console: print(line)

# ------------------------- Операции ФС -------------------------

@dataclass
class Summary:
    created_dirs: int = 0
    removed_dirs: int = 0
    removed_files: int = 0
    copied_files: int = 0
    replaced_files: int = 0
    verified: int = 0
    verify_failed: int = 0
    dry_run: bool = False

def ensure_dir(path: Path, dry: bool, log: List[str], s: Summary, to_console: bool) -> None:
    if not path.exists():
        if dry:
            log_append(log, f"[DRY] MKDIR {path}", to_console); s.created_dirs += 1; return
        _makedirs(path)
        log_append(log, f"[ OK ] MKDIR {path}", to_console); s.created_dirs += 1

def remove_file(path: Path, dry: bool, log: List[str], s: Summary, to_console: bool) -> None:
    if dry:
        log_append(log, f"[DRY] DELETE FILE {path}", to_console); s.removed_files += 1; return
    try:
        if is_windows():
            os.chmod(_win_long(path), stat.S_IWRITE)
            os.unlink(_win_long(path))
        else:
            path.unlink()
        log_append(log, f"[ OK ] DELETE FILE {path}", to_console); s.removed_files += 1
    except FileNotFoundError:
        log_append(log, f"[WARN] DELETE FILE not found {path}", to_console)
    except Exception as e:
        log_append(log, f"[FAIL] DELETE FILE {path} ({e})", to_console)

def remove_dir(path: Path, dry: bool, log: List[str], s: Summary, to_console: bool) -> None:
    if dry:
        log_append(log, f"[DRY] DELETE DIR  {path}", to_console); s.removed_dirs += 1; return
    try:
        shutil.rmtree(_win_long(path) if is_windows() else path)
        log_append(log, f"[ OK ] DELETE DIR  {path}", to_console); s.removed_dirs += 1
    except FileNotFoundError:
        log_append(log, f"[WARN] DELETE DIR not found {path}", to_console)
    except Exception as e:
        log_append(log, f"[FAIL] DELETE DIR {path} ({e})", to_console)

def copy_file(src: Path, dst: Path, dry: bool, log: List[str], s: Summary, to_console: bool) -> None:
    if dry:
        log_append(log, f"[DRY] COPY  {src} -> {dst}", to_console); s.copied_files += 1; return
    try:
        _makedirs(dst.parent)
        shutil.copy2(_win_long(src) if is_windows() else src,
                     _win_long(dst) if is_windows() else dst)
        log_append(log, f"[ OK ] COPY  {src} -> {dst}", to_console); s.copied_files += 1
    except Exception as e:
        log_append(log, f"[FAIL] COPY  {src} -> {dst} ({e})", to_console)

def replace_file(src: Path, dst: Path, dry: bool, log: List[str], s: Summary, to_console: bool, reason: str = "") -> None:
    if dry:
        log_append(log, f"[DRY] REPLACE {dst}  <-- {src} {reason}", to_console); s.replaced_files += 1; return
    try:
        _makedirs(dst.parent)
        shutil.copy2(_win_long(src) if is_windows() else src,
                     _win_long(dst) if is_windows() else dst)
        log_append(log, f"[ OK ] REPLACE {dst}  <-- {src} {reason}", to_console); s.replaced_files += 1
    except Exception as e:
        log_append(log, f"[FAIL] REPLACE {dst}  <-- {src} ({e})", to_console)

# ------------------------- Сравнение -------------------------

def need_replace_time_size(src: Path, dst: Path, time_field: str, eps: float, check_size: bool) -> bool:
    try:
        t_src = file_time(src, time_field); t_dst = file_time(dst, time_field)
    except FileNotFoundError:
        return True
    if abs(t_src - t_dst) > eps: return True
    if check_size:
        try:
            size_s = (os.stat(_win_long(src)).st_size if is_windows() else src.stat().st_size)
            size_d = (os.stat(_win_long(dst)).st_size if is_windows() else dst.stat().st_size)
            if size_s != size_d: return True
        except FileNotFoundError:
            return True
    return False

def equal_by_hash(src: Path, dst: Path, algo: str, chunk_mb: int) -> Optional[bool]:
    try:
        return hash_file(src, algo, chunk_mb) == hash_file(dst, algo, chunk_mb)
    except FileNotFoundError:
        return None

# ------------------------- Выбор путей -------------------------

def list_windows_drives() -> List[Tuple[str, str, str]]:
    import ctypes
    from ctypes import wintypes
    DRIVE_TYPES = {2: "Removable", 3: "Fixed", 4: "Remote", 5: "CD-ROM", 6: "RAMDisk"}
    drives = []
    bitmask = ctypes.windll.kernel32.GetLogicalDrives()
    for i in range(26):
        if bitmask & (1 << i):
            root = f"{chr(65 + i)}:\\"
            dtype = ctypes.windll.kernel32.GetDriveTypeW(wintypes.LPCWSTR(root))
            dtype_name = DRIVE_TYPES.get(dtype, "Unknown")
            vol_name_buf = ctypes.create_unicode_buffer(261)
            fs_name_buf = ctypes.create_unicode_buffer(261)
            serial = wintypes.DWORD(); max_comp = wintypes.DWORD(); flags = wintypes.DWORD()
            ctypes.windll.kernel32.GetVolumeInformationW(
                wintypes.LPCWSTR(root),
                vol_name_buf, len(vol_name_buf),
                ctypes.byref(serial), ctypes.byref(max_comp), ctypes.byref(flags),
                fs_name_buf, len(fs_name_buf),
            )
            label = vol_name_buf.value or ""
            drives.append((root, dtype_name, label))
    return drives

def pick_path_interactive(kind: str) -> Path:
    print(f"\nВыбор пути для {kind}:")
    if is_windows():
        drives = list_windows_drives()
        if drives:
            print("Найденные диски:")
            for idx, (root, dtype, label) in enumerate(drives, 1):
                lab = f" — {label}" if label else ""
                print(f"  {idx}. {root} ({dtype}{lab})")
            print("  0. Ввести путь вручную")
            while True:
                sel = input(f"Выберите номер диска для {kind} (или 0 для ввода пути): ").strip()
                if sel.isdigit():
                    n = int(sel)
                    if n == 0: break
                    if 1 <= n <= len(drives): return Path(drives[n-1][0])
                print("Некорректный выбор. Повторите.")
    while True:
        p = input(f"Введите путь к {kind} (например, E:\\ или C:\\Data\\USB): ").strip().strip('"')
        path = Path(p).expanduser()
        if path.exists(): return path
        print("Путь не существует. Повторите ввод.")

# ------------------------- CLI и финальная пауза -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Односторонняя синхронизация: флеш-диск (источник) -> папка (приёмник)")
    p.add_argument("--source","-s",type=str); p.add_argument("--target","-t",type=str)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--assume-yes", action="store_true")
    p.add_argument("--time-field", choices=["mtime","ctime"], default="mtime")
    p.add_argument("--time-epsilon", type=float, default=1.0)
    p.add_argument("--check-size", action="store_true")
    p.add_argument("--check-hash", choices=["never","if-needed","always","verify-after-copy"], default="if-needed")
    p.add_argument("--hash-algo", choices=["blake2b","sha256"], default="blake2b")
    p.add_argument("--hash-chunk-mb", type=int, default=8)
    p.add_argument("--log", type=str, default=None)
    p.add_argument("--progress", action="store_true")
    p.add_argument("--no-pause", action="store_true")
    p.add_argument("--version", action="store_true")
    return p.parse_args()

def confirm(prompt: str) -> bool:
    try: ans = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError: return False
    return ans in {"y","yes","д","да"}

def final_pause(enabled: bool) -> None:
    if not enabled: return
    print("\nСинхронизация завершена.")
    try:
        if is_windows():
            import msvcrt
            print("Нажмите любую клавишу для выхода...")
            msvcrt.getch()
        else:
            input("Нажмите Enter для выхода...")
    except Exception:
        time.sleep(8)

# ------------------------- Главная логика -------------------------

def main() -> None:
    args = parse_args()
    if args.version:
        print(f"one_way_sync.py version {VERSION}"); sys.exit(0)

    src = Path(args.source).expanduser() if args.source else pick_path_interactive("источника (флеш-диска)")
    dst = Path(args.target).expanduser() if args.target else pick_path_interactive("приёмника (папки на ПК)")

    validate_paths(src, dst)

    if not dst.exists() and not args.dry_run:
        _makedirs(dst)
        print(f"[ OK ] Создана папка-приёмник: {dst}")

    if args.time_field == "ctime" and not is_windows():
        print("Примечание: 'ctime' в этой ОС может не быть временем создания; используем birthtime, иначе mtime.", file=sys.stderr)

    print("Сканирование дерева файлов...")
    src_files, src_dirs = collect_tree(src)
    dst_files, dst_dirs = collect_tree(dst)

    # Индексы (Windows — без учёта регистра)
    SFi, SDi = index_by_key(src_files), index_by_key(src_dirs)
    TFi, TDi = index_by_key(dst_files), index_by_key(dst_dirs)

    # План
    dir_creates    = [SDi[k] for k in sorted(set(SDi) - set(TDi), key=lambda x: x.count('/'))]
    extra_dirs     = [TDi[k] for k in sorted(set(TDi) - set(SDi), key=lambda k: len(TDi[k].parts), reverse=True)]
    missing_files  = [SFi[k] for k in sorted(set(SFi) - set(TFi))]
    common_keys    = sorted(set(SFi) & set(TFi))
    extra_files    = [TFi[k] for k in sorted(set(TFi) - set(SFi))]

    # Подтверждение удаления
    do_delete = True
    destructive_count = len(extra_files) + len(extra_dirs)
    if destructive_count > 0 and not args.dry_run and not args.assume_yes:
        print(f"Найдено к удалению: {destructive_count} (файлов: {len(extra_files)}, папок: {len(extra_dirs)})")
        do_delete = confirm("Подтвердите удаление лишних объектов в приёмнике")
    if not do_delete:
        extra_files = []; extra_dirs = []

    total_steps = len(dir_creates) + len(missing_files) + len(common_keys) + len(extra_files) + len(extra_dirs)
    progress = Progress(total_steps, args.progress)
    console_verbose = not args.progress

    log: List[str] = []
    summary = Summary(dry_run=args.dry_run)

    # 1) Папки
    for drel in dir_creates:
        ensure_dir(dst / drel, args.dry_run, log, summary, console_verbose)
        progress.step(f"Папка: {drel}")

    # 2) Новые файлы
    for frel in missing_files:
        copy_file(src / frel, dst / frel, args.dry_run, log, summary, console_verbose)
        progress.step(f"Копия: {frel}")

    # 3) Общие файлы
    for key in common_keys:
        rel_src = SFi[key]; rel_dst = TFi[key]
        s = src / rel_src; d = dst / rel_dst

        replace = need_replace_time_size(s, d, args.time_field, args.time_epsilon, args.check_size)
        reason = ""
        if replace:
            try:
                ts_src = file_time(s, args.time_field); ts_dst = file_time(d, args.time_field)
                reason = f"(time {human_ts(ts_dst)} -> {human_ts(ts_src)})"
            except Exception:
                reason = "(time differs)"
        else:
            if args.check_hash in {"always","if-needed"}:
                eq = equal_by_hash(s, d, args.hash_algo, args.hash_chunk_mb)
                if eq is None: replace = True; reason = "(dest missing for hashing)"
                elif not eq:  replace = True; reason = "(hash differs)"
                else:         reason = "(hash equal)"

        if replace:
            replace_file(s, d, args.dry_run, log, summary, console_verbose, reason)
            if not args.dry_run and args.check_hash == "verify-after-copy":
                eq_after = equal_by_hash(s, d, args.hash_algo, args.hash_chunk_mb)
                if eq_after: log_append(log, f"[ OK ] VERIFY HASH {d}", console_verbose); summary.verified += 1
                else:        log_append(log, f"[FAIL] VERIFY HASH MISMATCH {d}", console_verbose); summary.verify_failed += 1

        progress.step(f"Проверка: {rel_src}")

    # 4) Удаления
    for frel in extra_files:
        remove_file(dst / frel, args.dry_run, log, summary, console_verbose)
        progress.step(f"Удаление файла: {frel}")
    for drel in extra_dirs:
        remove_dir(dst / drel, args.dry_run, log, summary, console_verbose)
        progress.step(f"Удаление папки: {drel}")

    progress.close()

    # 5) Итог
    print("\n=== ИТОГО ===")
    print(f"created_dirs : {summary.created_dirs}")
    print(f"removed_dirs : {summary.removed_dirs}")
    print(f"removed_files: {summary.removed_files}")
    print(f"copied_files : {summary.copied_files}")
    print(f"replaced_files: {summary.replaced_files}")
    if args.check_hash == "verify-after-copy":
        print(f"verify_ok    : {summary.verified}")
        print(f"verify_failed: {summary.verify_failed}")
    print(f"dry_run      : {summary.dry_run}")

    if args.log:
        try:
            Path(args.log).write_text(
                "\n".join(log) + "\n\nSUMMARY:\n" +
                "\n".join([
                    f"created_dirs : {summary.created_dirs}",
                    f"removed_dirs : {summary.removed_dirs}",
                    f"removed_files: {summary.removed_files}",
                    f"copied_files : {summary.copied_files}",
                    f"replaced_files: {summary.replaced_files}",
                    *( [f"verify_ok    : {summary.verified}", f"verify_failed: {summary.verify_failed}"]
                       if args.check_hash == "verify-after-copy" else [] ),
                    f"dry_run      : {summary.dry_run}",
                ]),
                encoding="utf-8"
            )
        except Exception as e:
            print(f"[WARN] Не удалось сохранить лог ({args.log}): {e}")

    # 6) Пост-проверка «хвостов»
    src_files2, src_dirs2 = collect_tree(src)
    dst_files2, dst_dirs2 = collect_tree(dst)
    SFi2, SDi2 = index_by_key(src_files2), index_by_key(src_dirs2)
    TFi2, TDi2 = index_by_key(dst_files2), index_by_key(dst_dirs2)
    leftovers_files = sorted(set(TFi2) - set(SFi2))
    leftovers_dirs  = sorted(set(TDi2) - set(SDi2))
    if leftovers_files or leftovers_dirs:
        print("\n[WARN] Обнаружены объекты, отсутствующие в источнике (не были удалены):")
        for k in leftovers_files[:50]: print("  файл :", TFi2[k])
        for k in leftovers_dirs[:50] : print("  папка:", TDi2[k])
        if len(leftovers_files) > 50 or len(leftovers_dirs) > 50:
            print("  ...список укорочен.")

    # 7) Финальная пауза
    final_pause(not args.no_pause)

if __name__ == "__main__":
    main()
