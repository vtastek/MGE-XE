using System;
using System.IO;
using System.Windows.Forms;
using SlimDX.Direct3D9;
using MGEgui.DirectX;
using MGEgui.DistantLand;

namespace MGEgui {
    /// <summary>
    /// Dev-only offline bake for the Forge viewer sky (SK-1). Reuses the dormant ProcessNif /
    /// BeginStaticCreation P/Invokes (Imports.cs) to bake the real MW sky meshes into the same
    /// static_meshes StaticElem blob the host already parses, plus extracts the clear-day cloud
    /// and sun textures as loose uncompressed DDS. One-time; regenerable from game files.
    ///
    /// Run:  MGEXEgui.exe --bake-sky   (from the Morrowind directory)
    /// Out:  Data Files\distantland\sky\sky_meshes
    ///       Data Files\distantland\sky\textures\tx_sky_clear.dds
    ///       Data Files\distantland\sky\textures\tx_sun_05.dds
    /// </summary>
    static class SkyBake {
        // Matches NifConverter.cpp StaticType. STATIC_NEAR skips the radius/cutoff cull that
        // STATIC_AUTO applies, so the dome/clouds always bake regardless of size.
        private const byte STATIC_NEAR = 1;

        private const string skyDir = @"Data Files\distantland\sky";
        private const string meshOut = skyDir + @"\sky_meshes";
        private const string texDir = skyDir + @"\textures";

        public static void Run() {
            // Offscreen D3D9 device on a hidden form (BeginStaticCreation needs a live device).
            Form host = new Form();
            host.Text = "MGE XE - Baking sky meshes...";
            host.ShowInTaskbar = false;
            host.FormBorderStyle = FormBorderStyle.FixedToolWindow;
            host.StartPosition = FormStartPosition.CenterScreen;
            host.Size = new System.Drawing.Size(320, 80);
            Label lbl = new Label { Text = "Baking sky meshes from game files...", Dock = DockStyle.Fill, TextAlign = System.Drawing.ContentAlignment.MiddleCenter };
            host.Controls.Add(lbl);
            host.Show();
            Application.DoEvents();

            try {
                DXMain.CheckAdapter();
                DXMain.CreateDevice(host);
                BSA.InitBSAs();

                Directory.CreateDirectory(skyDir);
                Directory.CreateDirectory(texDir);

                // --- Meshes: order is fixed so the host loader keys subsets by index ---
                //   index 0 = atmosphere dome (vertex-coloured, no texture)
                //   index 1 = clouds (single UV set, scrolled at runtime)
                // The atmosphere dome is texture-free (MW colours it per-weather at runtime); allow
                // textureless/UV-less shapes for the duration of this bake only.
                NativeMethods.SetAllowTexturelessShapes(1);
                NativeMethods.BeginStaticCreation(DXMain.device.ComPointer, Path.GetFullPath(meshOut));
                BakeNif("sky_atmosphere.nif");
                BakeNif("sky_clouds_01.nif");
                NativeMethods.EndStaticCreation();
                NativeMethods.SetAllowTexturelessShapes(0);

                // --- Textures: clear-day cloud sheet + sun billboard, uncompressed DDS ---
                ExtractDds("tx_sky_clear.tga", "tx_sky_clear.dds");
                ExtractDds("tx_sun_05.tga", "tx_sun_05.dds");

                MessageBox.Show("Sky bake complete:\n\n" + Path.GetFullPath(meshOut) + "\n" +
                                Path.GetFullPath(Path.Combine(texDir, "tx_sky_clear.dds")) + "\n" +
                                Path.GetFullPath(Path.Combine(texDir, "tx_sun_05.dds")),
                                "MGE XE - Bake Sky");
            } finally {
                try { BSA.CloseFiles(); } catch { }
                try { DXMain.CloseDevice(); } catch { }
                host.Close();
            }
        }

        private static void BakeNif(string nifName) {
            byte[] data = BSA.GetNif(nifName);
            // simplify=1.0 -> no progressive-mesh decimation (keep the dome intact).
            // cutoff is ignored for STATIC_NEAR.
            float r = NativeMethods.ProcessNif(data, data.Length, 1.0f, 0.0f, STATIC_NEAR);
            if (r < 0.0f) {
                throw new ApplicationException("ProcessNif failed for " + nifName + " (code " + r + ")");
            }
        }

        private static void ExtractDds(string srcName, string dstName) {
            byte[] data = BSA.GetTexture(srcName);

            // Work around a loading issue with TGA headers (matches DistantLandTextures.cs):
            // when ColorMapType == no_palette && ImageType == truecolor, zero ColorMapLength.
            if (data.Length > 6 && data[1] == 0 && (data[2] == 2 || data[2] == 10)) {
                data[5] = data[6] = 0;
            }

            // Uncompressed A8R8G8B8 so the host parseDds path is guaranteed to read it and alpha
            // (cloud coverage / sun disc) is preserved with no block-compression artefacts.
            Texture t = Texture.FromMemory(DXMain.device, data, 0, 0, 1, Usage.None,
                Format.A8R8G8B8, Pool.Scratch, Filter.None, Filter.None, 0);
            string outPath = Path.Combine(texDir, dstName);
            Texture.ToFile(t, outPath, ImageFileFormat.Dds);
            t.Dispose();
        }
    }
}
