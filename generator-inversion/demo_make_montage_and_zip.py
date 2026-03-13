import os, zipfile
from PIL import Image
import matplotlib.pyplot as plt

def show(ax, path, title):
    ax.imshow(Image.open(path), cmap="gray")
    ax.set_title(title, fontsize=9)
    ax.axis("off")

def main():
    work_dir = "geninv_demo"
    patch_dir = os.path.join(work_dir, "patch_demo")

    montage_path = os.path.join(work_dir, "montage.png")
    fig = plt.figure(figsize=(14, 8))
    axs = fig.subplots(2, 4).flatten()

    show(axs[0], os.path.join(work_dir, "Y_orig.png"), "Y orig")
    show(axs[1], os.path.join(work_dir, "Y_base.png"), "Y base (JPEG q50)")
    show(axs[2], os.path.join(work_dir, "err_base_vis.png"), "Error vis (orig-base)")
    show(axs[3], os.path.join(work_dir, "residual_Y_float_vis.png"), "Residual Y (float vis)")

    show(axs[4], os.path.join(patch_dir, "Y_orig_patch.png"), "Patch: Y orig")
    show(axs[5], os.path.join(patch_dir, "Y_base_patch.png"), "Patch: Y base")
    show(axs[6], os.path.join(patch_dir, "R_hat_vis.png"), "Patch: generated residual (vis)")
    show(axs[7], os.path.join(patch_dir, "Y_recon_patch.png"), "Patch: Y base + gen resid")

    fig.tight_layout()
    fig.savefig(montage_path, dpi=180)
    plt.close(fig)

    # Zip everything
    zip_path = "geninv_demo_outputs.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for root, _, files in os.walk(work_dir):
            for fn in files:
                full = os.path.join(root, fn)
                arc = os.path.relpath(full, work_dir)
                z.write(full, arcname=os.path.join("geninv_demo", arc))

    print("wrote:", montage_path)
    print("wrote:", zip_path)

if __name__ == "__main__":
    main()