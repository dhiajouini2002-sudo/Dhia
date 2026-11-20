# pip install opencv-python pillow ultralytics pytesseract
# + installer Tesseract OCR (avec langue "ara") et ajuster le chemin si Windows :
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import re, sqlite3, tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import cv2, pytesseract
from PIL import Image, ImageTk
from ultralytics import YOLO

model = YOLO("yolov8n.pt")          # détection voiture (classes COCO: 2,5,7)
db = sqlite3.connect("parking.db")
db.execute("""CREATE TABLE IF NOT EXISTS pointages(
    id INTEGER PRIMARY KEY AUTOINCREMENT, plaque TEXT,
    entree TEXT, sortie TEXT, duree TEXT)""")
FMT = "%Y-%m-%d %H:%M:%S"


# ---------- Fonctionnalité 1 : détection voiture ----------
def detecter_voiture(frame):
    res = model(frame, verbose=False)[0]
    best = None
    for b in res.boxes:
        if int(b.cls) in (2, 5, 7) and float(b.conf) > 0.4:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            if best is None or (x2-x1)*(y2-y1) > (best[2]-best[0])*(best[3]-best[1]):
                best = (x1, y1, x2, y2)
    if best:
        cv2.rectangle(frame, best[:2], best[2:], (0, 255, 0), 2)
    return best


# ---------- Fonctionnalité 2 : lecture plaque ----------
def localiser_plaque(car_img):
    gray = cv2.bilateralFilter(cv2.cvtColor(car_img, cv2.COLOR_BGR2GRAY), 11, 17, 17)
    edges = cv2.Canny(gray, 30, 200)
    cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:15]:
        x, y, w, h = cv2.boundingRect(c)
        if h > 0 and 2 < w / h < 6 and w > 60:
            return car_img[y:y+h, x:x+w]          # extraire la zone
    return car_img[int(car_img.shape[0]*0.6):, :]  # sinon bas de la voiture


def lire_plaque(plaque_img):
    g = cv2.cvtColor(plaque_img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # qualité
    txt = pytesseract.image_to_string(g, lang="ara+eng", config="--psm 7")
    nums = re.findall(r"\d+", txt)
    if len(nums) >= 2:
        return f"{nums[0]} تونس {nums[1]}"
    return " ".join(nums) if nums else ""


# ---------- Interface ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Parking Intelligent")
        self.cap, self.frame, self.plaque = None, None, ""
        self.video = tk.Label(self, width=640, height=360, bg="black")
        self.video.pack(padx=10, pady=10)
        self.lbl = tk.Label(self, text="Plaque : —", font=("Arial", 16, "bold"))
        self.lbl.pack()

        bar = tk.Frame(self); bar.pack(pady=5)
        for txt, cmd in [("Démarrer caméra", self.demarrer), ("Capturer", self.capturer),
                         ("Pointer Entrée", self.entree), ("Pointer Sortie", self.sortie)]:
            tk.Button(bar, text=txt, command=cmd, width=15).pack(side="left", padx=4)

        cols = ("plaque", "entree", "sortie", "duree")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=8)
        for c in cols:
            self.tree.heading(c, text=c.capitalize()); self.tree.column(c, width=150)
        self.tree.pack(padx=10, pady=10)
        self.maj()

    def demarrer(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.boucle()

    def boucle(self):
        if self.cap:
            ok, f = self.cap.read()
            if ok:
                self.frame = f
                img = cv2.cvtColor(cv2.resize(f, (640, 360)), cv2.COLOR_BGR2RGB)
                self.tkimg = ImageTk.PhotoImage(Image.fromarray(img))
                self.video.config(image=self.tkimg, width=640, height=360)
            self.after(30, self.boucle)

    def capturer(self):
        if self.frame is None:
            return messagebox.showwarning("Caméra", "Démarrez la caméra.")
        f = self.frame.copy()
        box = detecter_voiture(f)
        if not box:
            self.plaque = ""
            self.lbl.config(text="Aucune voiture détectée")
        else:
            car = f[box[1]:box[3], box[0]:box[2]]
            self.plaque = lire_plaque(localiser_plaque(car))
            self.lbl.config(text=f"Plaque : {self.plaque or 'illisible'}")
        img = cv2.cvtColor(cv2.resize(f, (640, 360)), cv2.COLOR_BGR2RGB)
        self.tkimg = ImageTk.PhotoImage(Image.fromarray(img))
        self.video.config(image=self.tkimg)
        if self.cap:                       # figer l'image capturée
            self.cap.release(); self.cap = None

    # ---------- Fonctionnalité 3 : pointage ----------
    def entree(self):
        if not self.plaque:
            return messagebox.showwarning("Entrée", "Aucune plaque lue.")
        deja = db.execute("SELECT 1 FROM pointages WHERE plaque=? AND sortie IS NULL",
                          (self.plaque,)).fetchone()
        if deja:
            return messagebox.showinfo("Entrée", "Véhicule déjà dans le parking.")
        db.execute("INSERT INTO pointages(plaque, entree) VALUES(?,?)",
                   (self.plaque, datetime.now().strftime(FMT)))
        db.commit(); self.maj()

    def sortie(self):
        if not self.plaque:
            return messagebox.showwarning("Sortie", "Aucune plaque lue.")
        row = db.execute("SELECT id, entree FROM pointages WHERE plaque=? AND sortie IS NULL "
                         "ORDER BY id DESC LIMIT 1", (self.plaque,)).fetchone()
        if not row:
            return messagebox.showinfo("Sortie", "Aucune entrée trouvée.")
        now = datetime.now()
        d = now - datetime.strptime(row[1], FMT)
        m = int(d.total_seconds() // 60)
        duree = f"{m // 60}h {m % 60}min"
        db.execute("UPDATE pointages SET sortie=?, duree=? WHERE id=?",
                   (now.strftime(FMT), duree, row[0]))
        db.commit(); self.maj()

    def maj(self):
        self.tree.delete(*self.tree.get_children())
        for r in db.execute("SELECT plaque, entree, COALESCE(sortie,'En cours'), "
                            "COALESCE(duree,'—') FROM pointages ORDER BY id DESC"):
            self.tree.insert("", "end", values=r)


if __name__ == "__main__":
    App().mainloop()
