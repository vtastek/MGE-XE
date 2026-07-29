using System;
using System.Windows.Forms;
using SlimDX;
using SlimDX.Direct3D9;
using TexCache = System.Collections.Generic.Dictionary<string, SlimDX.Direct3D9.Texture>;
using MGEgui.DistantLand;

namespace MGEgui.DirectX {
    public class LTEX {
        private static readonly TexCache texCache = new TexCache();

        public static void ReleaseCache() {
            foreach (Texture t in texCache.Values) {
                try {
                    if (!t.Disposed) {
                        t.Dispose();
                    }
                } catch {
                }
            }
            texCache.Clear();
        }

        private string filePath;
        public string FilePath {
            set { filePath = value.ToLowerInvariant(); }
            get { return filePath; }
        }
        public int index;
        public Texture tex;

        public void LoadTexture() {
            if (texCache.ContainsKey(filePath)) {
                tex = texCache[filePath];
            } else {
                try {
                    byte[] data = MGEgui.DistantLand.BSA.GetTexture(filePath);

                    // Work around a loading issue with TGA headers
                    // When ColorMapType == no_palette && ImageType == truecolor, set word ColorMapLength to 0
                    if (data.Length > 6 && data[1] == 0 && (data[2] == 2 || data[2] == 10)) {
                        data[5] = data[6] = 0;
                    }

                    // Scale down widest texture dimension to 256 to conserve memory
                    ImageInformation imginf = ImageInformation.FromMemory(data);
                    Filter dxf = Filter.Box | Filter.Srgb;
                    int maximal = Math.Max(imginf.Width, imginf.Height);
                    int skiplevels = 0, w = 0, h = 0;
                    while (maximal > 256) {
                        maximal /= 2;
                        skiplevels += 1;
                    }

                    if (skiplevels < imginf.MipLevels) {
                        // Skip loading mips larger than 256 pixels
                        // Filter bits 27-31 specify the number of mip levels to be skipped
                        dxf = (Filter)((skiplevels << 26) | (int)dxf);
                    } else {
                        // Rescale base level to fit inside 256 pixels
                        w = imginf.Width >> skiplevels;
                        h = imginf.Height >> skiplevels;
                    }

                    tex = Texture.FromMemory(DXMain.device, data, w, h, 0, Usage.None, Format.Unknown, Pool.Managed, Filter.Triangle | Filter.Dither, dxf, 0);
                    texCache[filePath] = tex;
                } catch {
                    tex = null;
                    throw;
                }
            }
        }
    }

    // Complete truncated mip chains on the LOOSE LAND textures (LTEX), reusing the exact repair
    // StaticTexCreator applies to statics source textures: append-only, authored mips kept
    // byte-for-byte, the untouched original backed up under distantland\mipfix_backup, BSA-packed
    // (vanilla) assets never rewritten. What is new is the driver — StaticTexCreator only ever sees
    // the statics texture set, and land textures are a disjoint set nothing was repairing.
    //
    // Why land textures need this at all: the Forge host now renders terrain from the REAL land
    // textures (tasks/forge-terrain.md), bucketed into Texture2DArrays. An array has ONE mip count
    // for every slice, so one source shipping a short chain caps every texture in its bucket —
    // measured on a 1024-heavy install, a single truncated DDS cut 463 land textures to 4 mips of
    // 11. Repairing the source is the only fix that helps the host AND MW's own near terrain, which
    // is why it belongs here rather than in the renderer.
    //
    // Idempotent: a repaired file has a complete chain and is skipped on every later run.
    static class LandMipFixer {
        public static int Checked, Fixed, SkippedBsa, SkippedComplete;
        public static readonly System.Collections.Generic.List<string> FixedPaths =
            new System.Collections.Generic.List<string>();
        public static readonly System.Collections.Generic.List<string> BsaSkippedPaths =
            new System.Collections.Generic.List<string>();

        public static void Reset() {
            Checked = Fixed = SkippedBsa = SkippedComplete = 0;
            FixedPaths.Clear();
            BsaSkippedPaths.Clear();
        }

        public static void Run(System.Collections.Generic.IEnumerable<string> texturePaths) {
            var seen = new System.Collections.Generic.HashSet<string>();
            foreach (string name in texturePaths) {
                if (string.IsNullOrEmpty(name) || !seen.Add(name.ToLowerInvariant())) {
                    continue;
                }
                try {
                    FixOne(name);
                } catch (Exception) {
                    // Unreadable or unrepairable is not fatal — the texture simply keeps the chain
                    // it shipped with, exactly as before this pass existed.
                }
            }
        }

        private static void FixOne(string name) {
            byte[] data = MGEgui.DistantLand.BSA.GetTexture(name);
            if (data == null || data.Length < 128) {
                return;
            }
            if (!(data[0] == 0x44 && data[1] == 0x44 && data[2] == 0x53 && data[3] == 0x20)) {
                return;   // not a DDS (a raw .tga) — nothing with a mip chain to complete
            }
            Checked++;

            int width = BitConverter.ToInt32(data, 16);
            int height = BitConverter.ToInt32(data, 12);
            int mipCount = Math.Max(1, BitConverter.ToInt32(data, 28));
            int pfFlags = BitConverter.ToInt32(data, 80);
            uint fourCC = BitConverter.ToUInt32(data, 84);
            int bitCount = BitConverter.ToInt32(data, 88);
            bool isCompressed = (pfFlags & 0x4) != 0;                                   // DDPF_FOURCC
            int blockSize = isCompressed ? ((fourCC == 0x31545844) ? 8 : 16) : (bitCount / 8);
            if (blockSize <= 0 || width <= 0 || height <= 0) {
                return;
            }

            int fullLevels = 1;
            for (int m = Math.Max(width, height); m > 1; m >>= 1) {
                fullLevels++;
            }
            if (mipCount >= fullLevels) {
                SkippedComplete++;
                return;
            }

            string loosePath = MGEgui.DistantLand.BSA.ResolveLoosePath(name);
            if (loosePath == null) {
                SkippedBsa++;                       // BSA-packed vanilla asset — report, never rewrite
                BsaSkippedPaths.Add(name);
                return;
            }
            if (StaticTexCreator.FixSourceMips(data, width, height, mipCount, fullLevels,
                                               blockSize, isCompressed, loosePath) != null) {
                Fixed++;
                FixedPaths.Add(loosePath);
            }
        }
    }

    class StaticTexCreator {
        private readonly System.Collections.Generic.HashSet<string> texCache;
        private readonly float pixelsPerWorld;
        private readonly int minLod;
        private readonly bool fixMips;

        // End-of-run report buckets
        public int Sliced = 0;
        public int Resampled = 0;
        public int Magenta = 0;
        public int MipFixed = 0;
        public int MipFixSkippedBsa = 0;
        public readonly System.Collections.Generic.List<string> ResampledPaths = new System.Collections.Generic.List<string>();
        public readonly System.Collections.Generic.List<string> MagentaPaths = new System.Collections.Generic.List<string>();
        public readonly System.Collections.Generic.List<string> MipFixedPaths = new System.Collections.Generic.List<string>();
        public readonly System.Collections.Generic.List<string> BsaSkippedPaths = new System.Collections.Generic.List<string>();

        // Backup directory for loose source textures the mip fixer rewrites (under distantland\, out
        // of MW's texture load path so the untouched original is never loaded, only kept to restore).
        public const string MipFixBackupDir = Statics.fn_dl + @"\mipfix_backup";

        // pixelsPerWorld = renderWidth / (2 * switchDistance * tan(hFov/2)) -- the on-screen texel
        // budget at the near->distant LOD switch. minLod = smallest texture dimension we ever emit.
        // (No max clamp: sizing never exceeds the source, and legit large textures are rare.)
        public StaticTexCreator(float pixelsPerWorld, int minLod, bool fixMips) {
            this.pixelsPerWorld = pixelsPerWorld;
            this.minLod = minLod;
            this.fixMips = fixMips;
            texCache = new System.Collections.Generic.HashSet<string>();
        }

        public void Dispose() {
            texCache.Clear();
        }

        private static int NextPow2(int v) {
            int p = 1;
            while (p < v) {
                p <<= 1;
            }
            return p;
        }

        // extent = world-space size (2 * bounding radius) of the LARGEST static using this texture.
        // The LOD texture is sized to the object's projected screen size at the switch, one mip
        // above (x2) for safety -- so the LOD can only ever be sharper than the near view.
        public bool LoadTexture(string path, float extent) {
            if (!texCache.Add(path)) {
                return true;
            }

            byte[] data = MGEgui.DistantLand.BSA.GetTexture(path);
            if (data == null) {
                return false;
            }

            // Work around a loading issue with TGA headers
            // When ColorMapType == no_palette && ImageType == truecolor, set word ColorMapLength to 0
            if (data.Length > 6 && data[1] == 0 && (data[2] == 2 || data[2] == 10)) {
                data[5] = data[6] = 0;
            }

            int pixelsAcross = (int)System.Math.Ceiling(extent * pixelsPerWorld);
            int targetDim = NextPow2(pixelsAcross) * 2;
            if (targetDim < minLod) {
                targetDim = minLod;
            }

            var outputPath = System.IO.Path.Combine(Statics.fn_stattex, System.IO.Path.ChangeExtension(path, ".dds"));
            System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(outputPath));

            // Non-DDS -> magenta placeholder so modders can see the asset isn't an optimized DDS.
            bool isDDS = data.Length >= 128 && data[0] == 0x44 && data[1] == 0x44 && data[2] == 0x53 && data[3] == 0x20;
            if (!isDDS) {
                if (WriteMagenta(outputPath, targetDim)) {
                    Magenta++;
                    MagentaPaths.Add(path);
                    return true;
                }
                return false;
            }

            // Parse the DDS header
            int width = BitConverter.ToInt32(data, 16);
            int height = BitConverter.ToInt32(data, 12);
            int mipCount = System.Math.Max(1, BitConverter.ToInt32(data, 28));
            int pfFlags = BitConverter.ToInt32(data, 80);
            uint fourCC = BitConverter.ToUInt32(data, 84);
            int bitCount = BitConverter.ToInt32(data, 88);
            bool isCompressed = (pfFlags & 0x4) != 0;                                  // DDPF_FOURCC
            int blockSize = isCompressed ? ((fourCC == 0x31545844) ? 8 : 16) : (bitCount / 8); // "DXT1"

            // Pick the source mip whose largest dimension is the smallest one still >= targetDim.
            int srcMax = System.Math.Max(width, height);
            int k = 0;
            while ((srcMax >> (k + 1)) >= targetDim && (srcMax >> (k + 1)) >= 4) {
                k++;
            }

            // A complete chain has floor(log2(maxDim)) + 1 levels (down to 1x1). Many modded source
            // DDS ship a short or absent chain; byte-copying that would leave the LOD texture with
            // missing mips, so only slice when the source really is complete -- otherwise fall
            // through to the decode+regenerate path, which rebuilds the whole pyramid.
            int fullLevels = 1;
            for (int m = srcMax; m > 1; m >>= 1) {
                fullLevels++;
            }
            bool completeChain = mipCount >= fullLevels;

            // The source ships a truncated mip chain -- the root defect behind both the slow
            // resample fallback below AND the near<->distant handoff mismatch (MW-near rebuilds the
            // missing tail from its truncated chain, the LOD from a fresh resample). If enabled,
            // complete the chain in place (loose files only) so BOTH derive from one full chain,
            // then fall into the fast byte-slice path for this run's LOD too.
            if (isDDS && !completeChain && fixMips) {
                string loosePath = MGEgui.DistantLand.BSA.ResolveLoosePath(path);
                if (loosePath == null) {
                    // BSA-only: vanilla asset, leave untouched -- just report it.
                    MipFixSkippedBsa++;
                    BsaSkippedPaths.Add(path);
                } else {
                    byte[] fixedBytes = FixSourceMips(data, width, height, mipCount, fullLevels, blockSize, isCompressed, loosePath);
                    if (fixedBytes != null) {
                        data = fixedBytes;
                        mipCount = System.Math.Max(1, BitConverter.ToInt32(data, 28));
                        completeChain = mipCount >= fullLevels;
                        MipFixed++;
                        MipFixedPaths.Add(path);
                    }
                }
            }

            // Fast path: slice mip k straight out of the source chain (keeps the author's filtering).
            if (completeChain && blockSize > 0 && k < mipCount) {
                int offset = 128, curW = width, curH = height, i = 0;
                bool ok = true;
                for (; i < k; i++) {
                    int mipSize = isCompressed
                        ? System.Math.Max(1, (curW + 3) / 4) * System.Math.Max(1, (curH + 3) / 4) * blockSize
                        : curW * curH * blockSize;
                    if (offset + mipSize > data.Length) { ok = false; break; }
                    offset += mipSize;
                    curW = System.Math.Max(1, curW / 2);
                    curH = System.Math.Max(1, curH / 2);
                }
                if (ok) {
                    int targetMipSize = isCompressed
                        ? System.Math.Max(1, (curW + 3) / 4) * System.Math.Max(1, (curH + 3) / 4) * blockSize
                        : curW * curH * blockSize;
                    if (offset + targetMipSize <= data.Length && WriteSlicedDDS(outputPath, data, offset, curW, curH, targetMipSize, mipCount - k)) {
                        Sliced++;
                        return true;
                    }
                }
            }

            // No / incomplete mips: decode the finest level present and downsample to targetDim.
            if (ResampleTexture(outputPath, data, targetDim)) {
                Resampled++;
                ResampledPaths.Add(path);
                return true;
            }
            return false;
        }

        // Copy the source's 128-byte header, retarget it so the chosen mip (w,h) becomes the top
        // level, and append that mip PLUS the rest of the source pyramid below it (everything from
        // offset to end already is that sub-chain). Keeps a real mip chain so the distant texture
        // doesn't shimmer as it recedes. Preserves the source pixel format exactly.
        private bool WriteSlicedDDS(string outputPath, byte[] data, int offset, int w, int h, int topMipSize, int mipLevels) {
            try {
                using (var fs = new System.IO.FileStream(outputPath, System.IO.FileMode.Create, System.IO.FileAccess.Write)) {
                    byte[] header = new byte[128];
                    Array.Copy(data, 0, header, 0, 128);
                    Array.Copy(BitConverter.GetBytes(h), 0, header, 12, 4);          // height
                    Array.Copy(BitConverter.GetBytes(w), 0, header, 16, 4);          // width
                    Array.Copy(BitConverter.GetBytes(topMipSize), 0, header, 20, 4); // pitch / linear size (top level)
                    Array.Copy(BitConverter.GetBytes(mipLevels), 0, header, 28, 4);  // mip count
                    Array.Copy(BitConverter.GetBytes(0x401008), 0, header, 108, 4);  // caps: TEXTURE | MIPMAP | COMPLEX
                    fs.Write(header, 0, 128);
                    fs.Write(data, offset, data.Length - offset);                    // top mip + rest of the pyramid
                }
                return true;
            } catch (System.IO.IOException) {
                return false;
            }
        }

        // Append-only mip completion for an incomplete LOOSE source DDS. Keeps every authored mip
        // byte-for-byte and only synthesises the MISSING tail below the smallest present level,
        // in the source's exact pixel format. Backs up the untouched original first, then rewrites
        // the loose file with a complete chain. Returns the fixed file bytes, or null on ANY failure
        // (leaving the original intact) so distant-land generation never breaks.
        // internal static so the LAND texture pass (LandMipFixer, below) can reuse it — the repair
        // is a property of the FILE, not of the statics bake, and land textures need it more:
        // the host renders terrain from bucketed Texture2DArrays, which have ONE mip count for all
        // slices, so a single short source caps the whole bucket. Uses no instance state.
        internal static byte[] FixSourceMips(byte[] data, int width, int height, int mipCount, int fullLevels, int blockSize, bool isCompressed, string loosePath) {
            Texture t = null;
            try {
                if (blockSize <= 0) {
                    return null;
                }

                // Walk the present mips down to the smallest authored level (same block-size loop as
                // the fast path). authoredEnd = byte offset just past the last authored mip.
                int offset = 128, curW = width, curH = height;
                int smallestOffset = 128, smallestW = width, smallestH = height, smallestSize = 0;
                for (int i = 0; i < mipCount; i++) {
                    int mipSize = isCompressed
                        ? System.Math.Max(1, (curW + 3) / 4) * System.Math.Max(1, (curH + 3) / 4) * blockSize
                        : curW * curH * blockSize;
                    if (offset + mipSize > data.Length) {
                        return null; // truncated / corrupt source
                    }
                    smallestOffset = offset;
                    smallestW = curW;
                    smallestH = curH;
                    smallestSize = mipSize;
                    offset += mipSize;
                    curW = System.Math.Max(1, curW / 2);
                    curH = System.Math.Max(1, curH / 2);
                }
                int authoredEnd = offset;

                // Preserve the source's exact D3D format (DXT1 -> DXT1, etc.).
                Format sourceFormat = ImageInformation.FromMemory(data).Format;

                // Wrap the smallest present mip as a standalone 1-level DDS, then let D3DX rebuild a
                // full chain from it: level 0 = that smallest present mip, levels 1.. = the tail.
                byte[] oneMip = new byte[128 + smallestSize];
                Array.Copy(data, 0, oneMip, 0, 128);
                Array.Copy(BitConverter.GetBytes(smallestH), 0, oneMip, 12, 4);        // height
                Array.Copy(BitConverter.GetBytes(smallestW), 0, oneMip, 16, 4);        // width
                Array.Copy(BitConverter.GetBytes(smallestSize), 0, oneMip, 20, 4);     // linear size
                Array.Copy(BitConverter.GetBytes(1), 0, oneMip, 28, 4);                // mip count
                Array.Copy(BitConverter.GetBytes(0x1000), 0, oneMip, 108, 4);          // caps: TEXTURE
                Array.Copy(data, smallestOffset, oneMip, 128, smallestSize);

                t = Texture.FromMemory(DXMain.device, oneMip, smallestW, smallestH, 0 /* full chain */, Usage.None, sourceFormat, Pool.Scratch, Filter.Box, Filter.Box, 0);

                byte[] gen;
                using (DataStream ds = Texture.ToStream(t, ImageFileFormat.Dds)) {
                    gen = new byte[ds.Length];
                    ds.Position = 0;
                    ds.Read(gen, 0, gen.Length);
                }

                // Skip the regenerated level 0 (identical dims/format to the smallest present mip)
                // and take only the appended tail.
                int genLevel0Size = isCompressed
                    ? System.Math.Max(1, (smallestW + 3) / 4) * System.Math.Max(1, (smallestH + 3) / 4) * blockSize
                    : smallestW * smallestH * blockSize;
                int tailStart = 128 + genLevel0Size;
                int tailLen = gen.Length - tailStart;
                if (tailLen <= 0) {
                    return null; // nothing to append -- treat as no-op rather than rewrite the file
                }

                // Compose: original header (mipCount = fullLevels, caps |= MIPMAP|COMPLEX, flags |=
                // MIPMAPCOUNT) + authored mips verbatim + synthesised tail.
                int authoredLen = authoredEnd - 128;
                byte[] fixedFile = new byte[128 + authoredLen + tailLen];
                Array.Copy(data, 0, fixedFile, 0, 128);
                int flags = BitConverter.ToInt32(data, 8) | 0x20000;                   // DDSD_MIPMAPCOUNT
                int caps = BitConverter.ToInt32(data, 108) | 0x401008;                 // TEXTURE | MIPMAP | COMPLEX
                Array.Copy(BitConverter.GetBytes(flags), 0, fixedFile, 8, 4);
                Array.Copy(BitConverter.GetBytes(fullLevels), 0, fixedFile, 28, 4);
                Array.Copy(BitConverter.GetBytes(caps), 0, fixedFile, 108, 4);
                Array.Copy(data, 128, fixedFile, 128, authoredLen);                    // authored mips, byte-for-byte
                Array.Copy(gen, tailStart, fixedFile, 128 + authoredLen, tailLen);     // appended tail

                // Back up the untouched original (never overwrite an existing backup -- keep the
                // true original), then rewrite the loose file. Mirror the sub-path under the source
                // root so backups never collide across subdirectories.
                string rel;
                if (loosePath.StartsWith(Statics.fn_textures + "\\", StringComparison.OrdinalIgnoreCase)) {
                    rel = loosePath.Substring(Statics.fn_textures.Length + 1);
                } else if (loosePath.StartsWith(Statics.fn_dataFiles + "\\", StringComparison.OrdinalIgnoreCase)) {
                    rel = loosePath.Substring(Statics.fn_dataFiles.Length + 1);
                } else {
                    rel = System.IO.Path.GetFileName(loosePath);
                }
                string backupPath = System.IO.Path.Combine(MipFixBackupDir, rel);
                System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(backupPath));
                if (!System.IO.File.Exists(backupPath)) {
                    System.IO.File.Copy(loosePath, backupPath);
                }
                System.IO.File.WriteAllBytes(loosePath, fixedFile);
                return fixedFile;
            } catch (Exception) {
                return null;
            } finally {
                if (t != null) {
                    t.Dispose();
                }
            }
        }

        // Decode the source (any format, incl. an incomplete chain) and re-emit a DDS at targetDim
        // (never upscaled beyond the source) with a freshly generated full mip chain, recompressed
        // to a sensible DXT format.
        private bool ResampleTexture(string outputPath, byte[] data, int targetDim) {
            ImageInformation imginfo;
            try {
                imginfo = ImageInformation.FromMemory(data);
            } catch (SlimDXException) {
                return false;
            }

            int newWidth = System.Math.Max(4, System.Math.Min(targetDim, imginfo.Width));
            int newHeight = System.Math.Max(4, System.Math.Min(targetDim, imginfo.Height));

            Format format;
            if (imginfo.Format == Format.Dxt1) {
                format = isDXT1a(imginfo, data) ? Format.Dxt3 : Format.Dxt1;
            } else if (imginfo.Format == Format.Dxt3 || imginfo.Format == Format.Dxt5) {
                format = imginfo.Format;
            } else if (imginfo.Format == Format.X8R8G8B8) {
                format = Format.Dxt1;
            } else {
                format = Format.Dxt3;
            }

            Texture t = null;
            try {
                t = Texture.FromMemory(DXMain.device, data, newWidth, newHeight, 0, Usage.None, format, Pool.Scratch, Filter.Triangle | Filter.Dither, Filter.Triangle, 0);
                Texture.ToFile(t, outputPath, ImageFileFormat.Dds);
                t.Dispose();
                return true;
            } catch (SlimDXException) {
                if (t != null) {
                    t.Dispose();
                }
                return false;
            }
        }

        // Solid-magenta DDS at (dim,dim) with a full mip chain. Used for non-DDS source textures.
        private bool WriteMagenta(string outputPath, int dim) {
            Texture t = null;
            try {
                t = new Texture(DXMain.device, dim, dim, 0, Usage.None, Format.A8R8G8B8, Pool.Scratch);
                DataRectangle dr = t.LockRectangle(0, LockFlags.None);
                byte[] row = new byte[dim * 4];
                for (int x = 0; x < dim; x++) {
                    row[x * 4 + 0] = 0xFF; // B
                    row[x * 4 + 1] = 0x00; // G
                    row[x * 4 + 2] = 0xFF; // R
                    row[x * 4 + 3] = 0xFF; // A
                }
                for (int y = 0; y < dim; y++) {
                    dr.Data.Seek((long)y * dr.Pitch, System.IO.SeekOrigin.Begin);
                    dr.Data.Write(row, 0, row.Length);
                }
                t.UnlockRectangle(0);
                t.FilterTexture(0, Filter.Point); // propagate magenta down the generated mip chain
                Texture.ToFile(t, outputPath, ImageFileFormat.Dds);
                t.Dispose();
                return true;
            } catch (SlimDXException) {
                if (t != null) {
                    t.Dispose();
                }
                return false;
            }
        }

        private bool isDXT1a(ImageInformation imginfo, byte[] data) {
            int blocks = (imginfo.Width * imginfo.Height) >> 4;

            for (int i = 0; i != blocks; ++i) {
                int k = 128 + 8 * i;
                uint c0 = (uint)(data[k + 0] | (data[k + 1] << 8));
                uint c1 = (uint)(data[k + 2] | (data[k + 3] << 8));
                uint b = (uint)(data[k + 4] | (data[k + 5] << 8) | (data[k + 6] << 16) | (data[k + 7] << 24));

                if (c0 <= c1 && ((b & 0x55555555) & (b >> 1)) != 0) {
                    return true;
                }
            }

            return false;
        }
    }

}
