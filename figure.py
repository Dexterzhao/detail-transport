"""Create the proposal figure from measures.csv and the same rest fields."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pilot import ROOT, CFG, relief


def font(size, bold=False):
    candidates = (("calibrib.ttf", "DejaVuSans-Bold.ttf") if bold
                  else ("calibri.ttf", "DejaVuSans.ttf"))
    for filename in candidates:
        for path in (Path("C:/Windows/Fonts")/filename, filename):
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                pass
    return ImageFont.load_default(size=size)


def main():
    with (ROOT/"measures.csv").open(newline="", encoding="utf-8") as stream:
        peak = {(r["asset"], r["policy"]): 100*float(r["detail_length_change_nrmse"])
                for r in csv.DictReader(stream) if float(r["amplitude"]) == 1}
    canvas = Image.new("RGB",(1390,570),"white")
    draw = ImageDraw.Draw(canvas)
    ink, muted, grid = "#182b40", "#415567", "#d4dee4"
    draw.text((25,10),"Rest relief",font=font(32,True),fill=ink)
    draw.text((355,10),"Relative RMS change in displacement-vector length (%)",
              font=font(32,True),fill=ink)
    colors = ["#34414d","#138477","#bc6038"]
    policies = list(CFG["policies"])
    names = ["Rotation only", "Square-root stretch", "Full gradient"]
    for x, color, name in zip((355,588,894),colors,names):
        draw.ellipse((x,67,x+15,82), fill=color)
        draw.text((x+23,57),name,font=font(27),fill=ink)
    x0, scale = 375, 20.4
    axis_y = 486
    for tick in range(0,46,5):
        x = int(x0 + tick*scale)
        draw.line((x,110,x,axis_y),fill=grid,width=2)
        draw.text((x,axis_y+5),str(tick),font=font(25),fill=muted,anchor="mt")
    draw.line((x0,axis_y,x0+45*scale,axis_y),fill=muted,width=2)
    # Each inset visualizes the entire [0,1]^2 rest UV chart with an identical
    # linear intensity scale for the normal displacement component.
    axis = np.linspace(0,1,100)
    xx, yy = np.meshgrid(axis,axis[::-1])
    pos=np.column_stack((xx.ravel(), yy.ravel(),np.zeros(xx.size)))
    for k, asset in enumerate(CFG["assets"]):
        y=165+125*k
        z=np.clip(relief(pos,asset)[:,2].reshape(xx.shape)/0.055,0,1)[:,:,None]
        white=np.array([243,241,233])
        dark=np.array([39,65,76])
        field=np.uint8(np.round(white+(dark-white)*z))
        im=Image.fromarray(field)
        canvas.paste(im,(28,y-50))
        draw.rectangle((28,y-50,127,y+49),outline="#9daeb3",width=1)
        title=asset.replace("_"," ").capitalize()
        draw.text((146,y),title,font=font(27,True),fill=ink,anchor="lm")
        for i, (policy, color) in enumerate(zip(policies,colors)):
            value=peak[asset,policy]
            end=int(x0+value*scale)
            yy0=y+26*(i-1)
            draw.line((x0,yy0,end,yy0),fill=color,width=4)
            draw.ellipse((end-6,yy0-6,end+6,yy0+6),fill=color)
            draw.text((end+12,yy0),f"{value:.2f}",font=font(23),fill=ink,anchor="lm")
    draw.text((25,532),"Inset shading shows the rest normal offset on the same 0–0.055 patch-width scale.",
              fill=muted,font=font(26))
    canvas.save(ROOT/"transport_check.png")


if __name__=="__main__":
    main()
