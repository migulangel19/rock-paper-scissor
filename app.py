"""
app.py
======

Punto de entrada unificado del Trabajo Final.

Flujo completo:

    1. Se abre una GUI (Tkinter) con dos opciones: **Iniciar sesión** o
       **Registrar usuario**. Reutiliza el módulo `login_facial` del
       compañero, basado en YuNet (detección) + SFace (embeddings) y
       comparación por similitud coseno.
    2. Cuando el usuario inicia sesión correctamente, se cierra la
       ventana de login y se lanza el juego de Piedra-Papel-Tijera
       (`juego.main(usuario=nombre)`), que muestra el nombre del
       jugador identificado en el HUD.
    3. Al salir del juego se vuelve al terminal.

Para ejecutarlo:

    python app.py
"""

from __future__ import annotations

import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path

import login_facial as lf

# Paleta de colores (idéntica a la del GUI original del compañero, para
# mantener la coherencia visual del proyecto integrado).
BG     = "#0d0d1a"
BG2    = "#1a1a2e"
GREEN  = "#00e676"
RED    = "#ff1744"
YELLOW = "#ffea00"
FG     = "#e0e0ff"
FG_DIM = "#6666aa"
MONO   = "Courier"
UI     = "Segoe UI"


class App(tk.Tk):
    """Ventana principal: login facial + arranque del juego."""

    def __init__(self):
        super().__init__()
        self.title("Piedra-Papel-Tijera con login facial")
        self.geometry("440x600")
        self.minsize(440, 600)            # no más pequeña que el diseño base
        self.resizable(True, True)        # se puede agrandar a gusto
        self.configure(bg=BG)
        self._centrar(440, 600)

        # Nombre del usuario identificado (None si nadie se loguea o se
        # cierra la ventana). Lo lee `main` tras salir del mainloop.
        self.usuario: str | None = None

        self._build()
        self._refresh_users()

        # Descarga de modelos en segundo plano la primera vez.
        if not lf.modelos_disponibles():
            self._status("Descargando modelos (primera vez)...", FG_DIM)
            self._busy(True)
            threading.Thread(target=self._descargar_modelos, daemon=True).start()

    # ---------------------------------------------------------------- UI

    def _centrar(self, w: int, h: int) -> None:
        """Coloca la ventana centrada (un poco por encima del centro)."""
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        x, y = (sw - w) // 2, max(0, (sh - h) // 3)
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _build(self):
        tk.Label(self, text="◈", font=(MONO, 38), bg=BG, fg=GREEN).pack(pady=(24, 0))
        tk.Label(self, text="PIEDRA · PAPEL · TIJERA",
                 font=(MONO, 13, "bold"), bg=BG, fg=GREEN).pack()
        tk.Label(self, text="Acceso por reconocimiento facial",
                 font=(UI, 9), bg=BG, fg=FG_DIM).pack(pady=(2, 14))

        self._line()

        tk.Label(self, text="NOMBRE  (solo para registrarse)",
                 font=(MONO, 9, "bold"), bg=BG, fg=FG_DIM).pack(pady=(16, 4))
        self.entry = tk.Entry(
            self, font=(MONO, 13), width=24,
            bg=BG2, fg=FG, insertbackground=GREEN, relief="flat", bd=0,
        )
        self.entry.pack(ipady=8, padx=50)
        self.entry.bind("<Return>", lambda _: self._login())

        c = tk.Canvas(self, height=2, bg=BG, highlightthickness=0)
        c.pack(fill="x", padx=50)
        c.create_line(0, 1, 500, 1, fill=GREEN, width=1)

        frame = tk.Frame(self, bg=BG)
        frame.pack(pady=24)

        self.btn_login = self._button(
            frame, "  INICIAR SESIÓN Y JUGAR  ",
            bg=GREEN, fg="#003320", command=self._login,
        )
        self.btn_login.pack(pady=6)

        self.btn_reg = self._button(
            frame, "  REGISTRAR USUARIO  ",
            bg=BG2, fg=GREEN, command=self._registrar, outlined=True,
        )
        self.btn_reg.pack(pady=6)

        self._line()

        tk.Label(self, text="USUARIOS REGISTRADOS",
                 font=(MONO, 8), bg=BG, fg=FG_DIM).pack(pady=(10, 2))
        self.lbl_users = tk.Label(self, text="—", font=(UI, 9),
                                  bg=BG, fg="#8888bb", wraplength=360)
        self.lbl_users.pack()

        self.lbl_status = tk.Label(self, text="", font=(MONO, 11, "bold"),
                                   bg=BG, fg=FG, wraplength=380, justify="center")
        self.lbl_status.pack(pady=14)

    def _line(self):
        tk.Frame(self, height=1, bg="#1e1e3a").pack(fill="x", padx=28)

    def _button(self, parent, text, bg, fg, command, outlined=False):
        btn = tk.Button(
            parent, text=text, font=(MONO, 10, "bold"),
            bg=bg, fg=fg, activebackground=bg, activeforeground=fg,
            relief="flat", cursor="hand2", width=26, pady=9, bd=0,
            highlightthickness=1 if outlined else 0,
            highlightbackground=fg if outlined else bg,
            command=command,
        )
        btn.bind("<Enter>", lambda _, b=btn, a=fg, c=bg: b.config(bg=a, fg=c))
        btn.bind("<Leave>", lambda _, b=btn, a=bg, c=fg: b.config(bg=a, fg=c))
        return btn

    # ------------------------------------------------------------ helpers

    def _status(self, msg, color=FG):
        self.lbl_status.config(text=msg, fg=color)

    def _busy(self, on):
        state = "disabled" if on else "normal"
        for w in (self.btn_login, self.btn_reg, self.entry):
            w.config(state=state)

    def _refresh_users(self):
        db = lf.cargar_db()
        self.lbl_users.config(text=", ".join(db.keys()) if db else "Ninguno")

    def _descargar_modelos(self):
        ok = lf.descargar_modelos()
        self.after(0, lambda: self._modelos_listos(ok))

    def _modelos_listos(self, ok):
        self._busy(False)
        if ok:
            self._status("Modelos listos.", GREEN)
        else:
            self._status("⚠  Error descargando modelos.\nRevisa tu conexión.", RED)

    # ------------------------------------------------------------- login

    def _login(self):
        db = lf.cargar_db()
        if not db:
            self._status("⚠  No hay usuarios registrados.\nRegistra uno primero.", YELLOW)
            return
        self._status("Abriendo cámara...", FG_DIM)
        self._busy(True)
        threading.Thread(target=self._run_login, daemon=True).start()

    def _run_login(self):
        try:
            nombre = lf.login()
        except Exception:
            import traceback
            traceback.print_exc()
            nombre = None
        self.after(0, lambda: self._done_login(nombre))

    def _done_login(self, nombre):
        self._busy(False)
        if nombre:
            self._status(f"✓  Bienvenido, {nombre}!  Lanzando juego...", GREEN)
            # Guardamos el nombre, salimos limpiamente del mainloop y dejamos
            # que `main()` lance el juego DESPUÉS de cerrar Tk. Si llamáramos
            # a juego.main() desde aquí dentro, el bucle de OpenCV bloquearía
            # el mainloop de Tk y la ventana del juego no se actualizaría.
            self.usuario = nombre
            # Pequeña pausa para que se vea el mensaje verde del status.
            self.after(150, self._salir_a_juego)
        else:
            self._status("✗  No reconocido", RED)

    def _salir_a_juego(self) -> None:
        self.destroy()

    # ---------------------------------------------------------- registro

    def _registrar(self):
        nombre = self.entry.get().strip()
        if not nombre:
            self._status("⚠  Introduce un nombre de usuario", YELLOW)
            return

        db = lf.cargar_db()
        if nombre in db:
            self._status(f"⚠  '{nombre}' ya existe.\nUsa 'Iniciar sesión' directamente.", YELLOW)
            return

        self._status("Abriendo cámara para registro...", FG_DIM)
        self._busy(True)
        threading.Thread(target=self._run_registro, args=(nombre,), daemon=True).start()

    def _run_registro(self, nombre: str):
        try:
            ok = lf.registrar(nombre)
        except Exception as e:
            print(f"Error registro: {e}")
            ok = False
        self.after(0, lambda: self._done_registro(nombre, ok))

    def _done_registro(self, nombre: str, ok: bool):
        self._busy(False)
        self._refresh_users()
        if ok:
            self._status(
                f"✓  Usuario '{nombre}' registrado.\nYa puedes iniciar sesión.",
                GREEN,
            )
        else:
            self._status("Registro cancelado.", FG_DIM)


def main() -> None:
    app = App()
    app.mainloop()                       # bloquea hasta que se cierre el GUI
    if app.usuario:                       # solo si hubo login correcto
        # Lanzamos el juego como subproceso para que tenga un intérprete
        # limpio (Tk y OpenCV/Qt no se llevan bien en el mismo proceso:
        # tras destruir la ventana de Tk, la siguiente `cv2.imshow`
        # se queda colgada).
        juego_path = Path(__file__).parent / "juego.py"
        subprocess.run([sys.executable, str(juego_path), app.usuario])


if __name__ == "__main__":
    main()
