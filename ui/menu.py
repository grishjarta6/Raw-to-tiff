"""Интерактивное меню и основной цикл обработки."""

from pathlib import Path

from core.mxf_parser import parse as mxf_parse
from core.autodetect import autodetect
from core.decoder import (make_decoder, split_channels,
                          detect_resolution)
from core.stats import print_detection, print_channels_short
from core.exporter import save_frame


AUTO = 0


def find_raw_files(directory: Path) -> list[Path]:
    found = set()
    for ext in ("*.mxf", "*.MXF", "*.ari", "*.ARI"):
        found.update(directory.glob(ext))
    return sorted(found)


def _resolve_resolution(filepath: Path, W: int, H: int,
                        result: dict) -> tuple[int, int, str]:
    essence_size = result["frame_size"]

    if W > 0 and H > 0:
        return W, H, "вручную"

    cands = detect_resolution(essence_size)
    if not cands:
        raise RuntimeError(
            f"Не могу определить разрешение для essence_size="
            f"{essence_size:,}. Задайте вручную (пункт [p])."
        )

    if len(cands) > 1:
        print(f"   ⚠ Найдено несколько разрешений:")
        for i, (w, h, hdr, name) in enumerate(cands, 1):
            print(f"      [{i}] {w}×{h}  header={hdr}  {name}")
        print(f"   ℹ Использую первое.")

    W_, H_, header, name = cands[0]
    return W_, H_, f"авто ({name}, header={header})"


def process_file(filepath, W, H, force_params=None, verbose=False,
                 frame_mode="all", frame_index=0, engine="auto") -> None:
    print(f"\n{'═' * 72}")
    print(f"📂 {filepath.name}")
    print(f"{'═' * 72}")

    # --- Fallback принудительно
    if engine == "fallback":
        print(f"\n   🔁 Движок: fallback (принудительно)")
        from core.fallback_engine import try_process
        out_dir = filepath.parent / f"{filepath.stem}_channels"
        ok = try_process(filepath, out_dir=out_dir, save_rgb=True)
        if not ok:
            print(f"   ❌ Fallback не сработал.")
        return

    # --- MXF
    try:
        result = mxf_parse(filepath)
    except Exception as e:
        print(f"   ❌ Ошибка парсинга MXF: {e}")
        if engine == "auto":
            print(f"\n   🔁 Пробую fallback-движок...")
            from core.fallback_engine import try_process
            out_dir = filepath.parent / f"{filepath.stem}_channels"
            ok = try_process(filepath, out_dir=out_dir, save_rgb=True)
            if not ok:
                print(f"   ❌ Fallback тоже не сработал.")
        else:
            print(f"   ℹ Движок=mxf, fallback отключён.")
        return

    packets = result["essence"]
    print(f"   Кадров:         {len(packets)}")
    print(f"   Размер пакета:  {packets[0]['length']:,} байт")

    try:
        W_use, H_use, src = _resolve_resolution(filepath, W, H, result)
    except Exception as e:
        print(f"   ❌ {e}")
        return
    if (W_use, H_use) != (W, H) or W == AUTO or H == AUTO:
        print(f"   📐 Разрешение:  {W_use} × {H_use}  ({src})")

    # --- Параметры распаковки
    if force_params:
        det = dict(force_params)
        print(f"\n   ℹ Ручные параметры:")
        for k, v in force_params.items():
            print(f"      {k}: {v}")
    else:
        print(f"\n   🔍 Автоопределение параметров...")
        det = autodetect(filepath, packets[0], W_use, H_use, verbose=verbose)

        # Если det плох и разрешён auto — переключаемся на fallback
        det_bad = (det is None) or (det.get("corr_g1g2", 0) < 0.5)
        if det_bad and engine == "auto":
            print(f"\n   ⚠ MXF-движок не дал уверенного результата "
                  f"(corr_G = {det['corr_g1g2'] if det else 'н/д'}).")
            print(f"   🔁 Переключаюсь на fallback-движок...")
            from core.fallback_engine import try_process
            out_dir = filepath.parent / f"{filepath.stem}_channels"
            ok = try_process(filepath, out_dir=out_dir, save_rgb=True)
            if not ok:
                print(f"   ❌ Fallback тоже не сработал.")
            return

        if det is None:
            print(f"   ❌ Не удалось определить параметры.")
            return
        print_detection(det)

    # --- Собираем декодер из det
    try:
        decode_raw = make_decoder(det)
    except Exception as e:
        print(f"   ❌ Не удалось создать декодер: {e}")
        return

    # --- Какие кадры
    if frame_mode == "first":
        indices = [0]
    elif frame_mode == "one":
        if not (0 <= frame_index < len(packets)):
            print(f"   ❌ Кадр {frame_index} вне диапазона "
                  f"(0..{len(packets) - 1})")
            return
        indices = [frame_index]
    else:
        indices = list(range(len(packets)))

    out_dir = filepath.parent / f"{filepath.stem}_channels"
    pattern = det["pattern"]

    mode_label = {"all": f"все ({len(indices)})",
                  "first": "первый",
                  "one": f"кадр {frame_index}"}[frame_mode]

    print(f"\n   📦 Режим: {mode_label}")
    print(f"   📦 Сохраняю в: {out_dir}")
    print()

    for idx in indices:
        pkt = packets[idx]
        try:
            with open(filepath, "rb") as f:
                f.seek(pkt["value_start"])
                raw = f.read(pkt["length"])

            frame = decode_raw(raw, W_use, H_use)
            chans = split_channels(frame, pattern)

            save_frame(chans, out_dir, idx, pattern, save_rgb=True)
            print(f"   ✅ кадр {idx:02d} → {print_channels_short(chans)}")
        except Exception as e:
            print(f"   ❌ кадр {idx:02d}: {type(e).__name__}: {e}")

    print(f"\n   ✅ Готово: {out_dir}")


# ===========================================================================
# Меню
# ===========================================================================

def _prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{msg}{suffix}: ").strip()
    return val if val else default


def _prompt_int(msg: str, default: int) -> int:
    while True:
        raw = input(f"{msg} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("   ❌ Нужно число")


def _prompt_frames(total: int, current_mode: str,
                   current_idx: int) -> tuple[str, int]:
    print(f"\n  Какие кадры обрабатывать?")
    print(f"     [1] только первый кадр")
    print(f"     [2] все кадры ({total})")
    print(f"     [3] конкретный кадр")
    choice = input("  Выбор [2]: ").strip() or "2"
    if choice == "1":
        return "first", 0
    if choice == "2":
        return "all", 0
    if choice == "3":
        idx = _prompt_int(f"     Номер кадра (0..{total - 1})", current_idx)
        return "one", idx
    print("   ❌ Неверный выбор, оставляю как было")
    return current_mode, current_idx


def main_menu() -> None:
    cwd = Path.cwd()
    W, H = AUTO, AUTO
    manual_params: dict | None = None
    verbose = False
    frame_mode = "all"
    frame_index = 0
    engine = "auto"

    while True:
        files = find_raw_files(cwd)
        mode_label = {"all": "все кадры",
                      "first": "первый кадр",
                      "one": f"кадр {frame_index}"}[frame_mode]
        res_label = "авто" if (W == AUTO or H == AUTO) else f"{W}×{H}"

        print()
        print("═" * 72)
        print(f"  ARRIRAW → TIFF  (ALEXA Mini / LF)")
        print(f"  Папка:          {cwd}")
        print(f"  Разрешение:     {res_label}")
        print(f"  Движок:         {engine}")
        print(f"  Режим расп.:    "
              f"{'ручной: ' + str(manual_params) if manual_params else 'авто'}")
        print(f"  Кадры:          {mode_label}")
        print(f"  Verbose:        {verbose}")
        print("═" * 72)

        if files:
            print(f"\n  📁 Файлы ({len(files)}):")
            for i, f in enumerate(files, 1):
                mb = f.stat().st_size / (1024 * 1024)
                print(f"     [{i:>2}] {f.name}  ({mb:>8.1f} МБ)")
        else:
            print(f"\n  ❌ В папке нет .mxf/.ari файлов")

        print(f"\n  Действия:")
        if files:
            print(f"     [1..{len(files)}]  обработать один файл")
            print(f"     [a]          обработать все")
        print(f"     [e]          движок: авто/mxf/fallback (сейчас: {engine})")
        print(f"     [f]          какие кадры (сейчас: {mode_label})")
        print(f"     [d]          сменить папку")
        print(f"     [r]          обновить список")
        print(f"     [p]          разрешение (сейчас: {res_label})")
        print(f"     [m]          режим распаковки: авто/ручной")
        print(f"     [v]          verbose вкл/выкл")
        print(f"     [q]          выход")

        choice = input("\n  Выбор: ").strip().lower()

        if choice == "q":
            print("  Выход.")
            return
        elif choice == "r":
            continue
        elif choice == "f":
            total = 1
            if files:
                try:
                    r = mxf_parse(files[0])
                    total = len(r["essence"])
                except Exception:
                    pass
            frame_mode, frame_index = _prompt_frames(total, frame_mode,
                                                     frame_index)
        elif choice == "d":
            newdir = _prompt("  Путь к папке").strip()
            p = Path(newdir).expanduser().resolve()
            if p.is_dir():
                cwd = p
            else:
                print(f"   ❌ Не папка: {newdir}")
        elif choice == "p":
            print(f"\n  Разрешение кадра:")
            print(f"     [1] авто")
            print(f"     [2] вручную")
            sub = input("  Выбор [1]: ").strip() or "1"
            if sub == "2":
                W = _prompt_int("     Ширина", W if W else 3424)
                H = _prompt_int("     Высота", H if H else 2202)
            else:
                W, H = AUTO, AUTO
                print(f"   ✅ Разрешение: авто")
        elif choice == "v":
            verbose = not verbose
            print(f"   Verbose: {verbose}")
        elif choice == "m":
            print(f"\n  Режим распаковки:")
            print(f"     [1] автоопределение")
            print(f"     [2] ручной")
            sub = input("  Выбор [1]: ").strip() or "1"
            if sub == "2":
                manual_params = _prompt_manual_params()
                print(f"   ✅ Ручные параметры: {manual_params}")
            else:
                manual_params = None
                print(f"   ✅ Режим: автоопределение")
        elif choice == "e":
            print(f"\n  Движок распаковки:")
            print(f"     [1] auto — MXF, если не сработал → fallback")
            print(f"     [2] mxf — только MXF-парсер")
            print(f"     [3] fallback — только чтение как изображение")
            sub = input("  Выбор [1]: ").strip() or "1"
            engine = {"1": "auto", "2": "mxf", "3": "fallback"}.get(sub, "auto")
            print(f"   ✅ Движок: {engine}")
        elif choice == "a" and files:
            for f in files:
                try:
                    process_file(f, W, H,
                                 force_params=manual_params,
                                 verbose=verbose,
                                 frame_mode=frame_mode,
                                 frame_index=frame_index,
                                 engine=engine)
                except KeyboardInterrupt:
                    print("\n   ⏹ Прервано пользователем")
                    break
                except Exception as e:
                    print(f"   ❌ {f.name}: {type(e).__name__}: {e}")
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    process_file(files[idx], W, H,
                                 force_params=manual_params,
                                 verbose=verbose,
                                 frame_mode=frame_mode,
                                 frame_index=frame_index,
                                 engine=engine)
                else:
                    print("   ❌ Неверный номер")
            except ValueError:
                print("   ❌ Неверный выбор")


def _prompt_manual_params() -> dict:
    from core.decoder import (UNPACK_FUNCS, ASSEMBLIES, PATTERNS as P,
                              SIDE_FORMULAS)

    print("\n  Ручные параметры распаковки:")
    header = _prompt_int("     header", 76)

    print(f"\n     Вариант A: каноническое имя")
    print(f"       {list(UNPACK_FUNCS)}")
    print(f"     Вариант B: пара индексов i1 / i2 из SIDE_FORMULAS "
          f"(0..{len(SIDE_FORMULAS) - 1})")

    use_pair = input("     Использовать пару (i1,i2)? [n]: ").strip().lower()
    result = {"header": header}

    if use_pair == "y":
        result["i1"] = _prompt_int("     i1", 2)
        result["i2"] = _prompt_int("     i2", 7)
        result["unpack"] = f"pair({result['i1']},{result['i2']})"
    else:
        unpack = _prompt("     unpack", "arri_alt")
        if unpack not in UNPACK_FUNCS:
            unpack = "arri_alt"
        result["unpack"] = unpack

    print(f"     assembly:  {list(ASSEMBLIES)}")
    assembly = _prompt("     assembly", "top_bottom")
    if assembly not in ASSEMBLIES:
        assembly = "top_bottom"
    result["assembly"] = assembly

    print(f"     pattern:   {list(P)}")
    pattern = _prompt("     pattern", "GBRG")
    if pattern not in P:
        pattern = "GBRG"
    result["pattern"] = pattern

    return result