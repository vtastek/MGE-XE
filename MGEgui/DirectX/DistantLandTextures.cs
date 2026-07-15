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
        private byte[] FixSourceMips(byte[] data, int width, int height, int mipCount, int fullLevels, int blockSize, bool isCompressed, string loosePath) {
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

    class TextureBank {
        public LTEX t1, t2, t3, t4;
        public VertexBuffer wBuffer;

        public struct WeightVertex {
            public byte w1, w2, w3, w4;

            public const VertexFormat Format = VertexFormat.Diffuse;
            public const int Stride = 4;
        };

        public TextureBank() {
            wBuffer = new VertexBuffer(DXMain.device, WeightVertex.Stride * 4225, Usage.WriteOnly, WeightVertex.Format, Pool.Managed);
        }

        public void SetSingleTexture(LTEX tex1) {
            // There this will create a default bank which can be used for default land which has one texture
            t1 = tex1;

            DataStream WeightData = wBuffer.Lock(0, 0, LockFlags.None);
            WeightVertex defaultw;
            defaultw.w1 = 255;
            defaultw.w2 = 0;
            defaultw.w3 = 0;
            defaultw.w4 = 0;

            for (int y = 0; y <= 64; y++) {
                for (int x = 0; x <= 64; x++) {
                    WeightData.Write(defaultw);
                }
            }

            wBuffer.Unlock();
        }

        private WeightVertex SampleWeightData(ref WeightVertex[] array, int x, int y) {

            // Ensure that x and y do not escape the bounds of the array.
            if (x < 0) {
                x = 0;
            }
            if (y < 0) {
                y = 0;
            }
            if (x > 64) {
                x = 64;
            }
            if (y > 64) {
                y = 64;
            }

            // Return the value at the constrained location
            return array[y * 65 + x];
        }
        public void CalcWeights(LAND cell) {

            var WeightData = new WeightVertex[65 * 65];

            for (int y = 0; y <= 64; y++) {
                for (int x = 0; x <= 64; x++) {
                    // Figure out which index to use
                    int i = y * 65 + x;

                    // Figure out which texture is used here, match Morrowind rounding
                    int cell_x = cell.xpos;
                    int cell_y = cell.ypos;
                    int tex_x = (int)Math.Floor(((float)x - 2.0f) / 4.0f);
                    int tex_y = (int)Math.Ceiling(((float)y - 2.0f) / 4.0f);

                    DistantLandForm.ModCell(ref cell_x, ref tex_x);
                    DistantLandForm.ModCell(ref cell_y, ref tex_y);

                    LTEX tmp = DistantLandForm.GetTex(cell_x, cell_y, tex_x, tex_y);
                    string tex_index = tmp.FilePath;

                    // Write values
                    if (t1 != null && t1.FilePath == tex_index) {
                        WeightData[i].w1 = 255;
                        continue;
                    } else {
                        WeightData[i].w1 = 0;
                    }

                    if (t2 != null && t2.FilePath == tex_index) {
                        WeightData[i].w2 = 255;
                        continue;
                    } else {
                        WeightData[i].w2 = 0;
                    }

                    if (t3 != null && t3.FilePath == tex_index) {
                        WeightData[i].w3 = 255;
                        continue;
                    } else {
                        WeightData[i].w3 = 0;
                    }

                    if (t4 != null && t4.FilePath == tex_index) {
                        WeightData[i].w4 = 255;
                        continue;
                    } else {
                        WeightData[i].w4 = 0;
                    }
                }
            }

            // Blur the weights as we transfer them so the transitions aren't quite so blocky and horrible.
            // Blur kernel
            var blur = new float[] { 0.04f, 0.16f, 0.6f, 0.16f, 0.04f };

            // Horizontal Pass
            var FirstPassWD = new WeightVertex[65 * 65];
            for (int y = 0; y <= 64; y++) {
                for (int x = 0; x <= 64; x++) {
                    // Figure out which index to use
                    int i = y * 65 + x;

                    if (x == 0 || x == 64 || y == 0 || y == 64) {
                        // We're at the edge, so just copy the value (don't want to interfere with the way the edges of cells look
                        FirstPassWD[i] = WeightData[i];
                        continue;
                    }

                    // We're not at the edge, so add some influence from the surrounding weights
                    // Additional incides
                    WeightVertex wv0, wv1, wv2, wv3, wv4;
                    float value;

                    wv0 = SampleWeightData(ref WeightData, x - 2, y);
                    wv1 = SampleWeightData(ref WeightData, x - 1, y);
                    wv2 = SampleWeightData(ref WeightData, x, y);
                    wv3 = SampleWeightData(ref WeightData, x + 1, y);
                    wv4 = SampleWeightData(ref WeightData, x + 2, y);

                    value = (float)wv0.w1 * blur[0] + (float)wv1.w1 * blur[1] + (float)wv2.w1 * blur[2] + (float)wv3.w1 * blur[3] + (float)wv4.w1 * blur[4];
                    FirstPassWD[i].w1 = (byte)value;

                    value = (float)wv0.w2 * blur[0] + (float)wv1.w2 * blur[1] + (float)wv2.w2 * blur[2] + (float)wv3.w2 * blur[3] + (float)wv4.w2 * blur[4];
                    FirstPassWD[i].w2 = (byte)value;

                    value = (float)wv0.w3 * blur[0] + (float)wv1.w3 * blur[1] + (float)wv2.w3 * blur[2] + (float)wv3.w3 * blur[3] + (float)wv4.w3 * blur[4];
                    FirstPassWD[i].w3 = (byte)value;

                    value = (float)wv0.w4 * blur[0] + (float)wv1.w4 * blur[1] + (float)wv2.w4 * blur[2] + (float)wv3.w4 * blur[3] + (float)wv4.w4 * blur[4];
                    FirstPassWD[i].w4 = (byte)value;
                }
            }

            // Vertical pass - writes to final vertex buffer
            DataStream FinalWeightData = wBuffer.Lock(0, 0, LockFlags.None);

            // Blur the weights as we transfer them so the transitions aren't quite so blocky and horrible.
            for (int y = 0; y <= 64; y++) {
                for (int x = 0; x <= 64; x++) {
                    if (x == 0 || x == 64 || y == 0 || y == 64) {
                        // We're at the edge, so just copy the value (don't want to interfere with the way the edges of cells look
                        FinalWeightData.Write(WeightData[65 * y + x]);
                        continue;
                    }

                    // We're not at the edge, so add some influence from the surrounding weights
                    // Additional incides
                    WeightVertex wv0, wv1, wv2, wv3, wv4, wvfinal;
                    float value;
                    
                    wv0 = SampleWeightData(ref FirstPassWD, x, y - 2);
                    wv1 = SampleWeightData(ref FirstPassWD, x, y - 1);
                    wv2 = SampleWeightData(ref FirstPassWD, x, y);
                    wv3 = SampleWeightData(ref FirstPassWD, x, y + 1);
                    wv4 = SampleWeightData(ref FirstPassWD, x, y + 2);

                    value = (float)wv0.w1 * blur[0] + (float)wv1.w1 * blur[1] + (float)wv2.w1 * blur[2] + (float)wv3.w1 * blur[3] + (float)wv4.w1 * blur[4];
                    wvfinal.w1 = (byte)value;

                    value = (float)wv0.w2 * blur[0] + (float)wv1.w2 * blur[1] + (float)wv2.w2 * blur[2] + (float)wv3.w2 * blur[3] + (float)wv4.w2 * blur[4];
                    wvfinal.w2 = (byte)value;

                    value = (float)wv0.w3 * blur[0] + (float)wv1.w3 * blur[1] + (float)wv2.w3 * blur[2] + (float)wv3.w3 * blur[3] + (float)wv4.w3 * blur[4];
                    wvfinal.w3 = (byte)value;

                    value = (float)wv0.w4 * blur[0] + (float)wv1.w4 * blur[1] + (float)wv2.w4 * blur[2] + (float)wv3.w4 * blur[3] + (float)wv4.w4 * blur[4];
                    wvfinal.w4 = (byte)value;

                    FinalWeightData.Write(wvfinal);
                }
            }

            wBuffer.Unlock();
        }

        ~TextureBank() {
            wBuffer.Dispose();
        }

    };

    class CellTexCreator {
        private struct CellVertex {
            public float x, y, z, w;
            public float u, v;

            public const VertexFormat Format = VertexFormat.Position | VertexFormat.Texture1;
            public const int Stride = 24;
        };

        private struct NormalColorVertex {
            public float nx, ny, nz;
            public byte b, g, r, a;

            public const VertexFormat Format = VertexFormat.Normal | VertexFormat.Diffuse;
            public const int Stride = 16;
        };

        VertexElement[] Elements = new VertexElement[] {
            // Stream 0 - position and texture coordinates
            new VertexElement(0, 0, DeclarationType.Float4, DeclarationMethod.Default, DeclarationUsage.Position, 0),
            new VertexElement(0, 16, DeclarationType.Float2, DeclarationMethod.Default, DeclarationUsage.TextureCoordinate, 0),

            // Stream 1 - normals and vertex colors
            new VertexElement(1, 0, DeclarationType.Float3, DeclarationMethod.Default, DeclarationUsage.Normal, 0),
            new VertexElement(1, 12, DeclarationType.Color, DeclarationMethod.Default, DeclarationUsage.Color, 0),

            // Stream 2 - Texture weights
            new VertexElement(2, 0, DeclarationType.Color, DeclarationMethod.Default, DeclarationUsage.Color, 1),

            VertexElement.VertexDeclarationEnd
        };

        VertexElement[] NormalElements = new VertexElement[] {
            // Stream 0 - position and texture coordinates
            new VertexElement(0, 0, DeclarationType.Float4, DeclarationMethod.Default, DeclarationUsage.Position, 0),
            new VertexElement(0, 16, DeclarationType.Float2, DeclarationMethod.Default, DeclarationUsage.TextureCoordinate, 0),

            // Stream 1 - normals and vertex colors
            new VertexElement(1, 0, DeclarationType.Float3, DeclarationMethod.Default, DeclarationUsage.Normal, 0),
            new VertexElement(1, 12, DeclarationType.Color, DeclarationMethod.Default, DeclarationUsage.Color, 0),

            VertexElement.VertexDeclarationEnd
        };

        private const string EffectPath = @"Data Files\shaders\core\CellTexBlend.fx";
        private VertexBuffer vBuffer;
        private VertexBuffer colorBuffer;
        private System.Collections.Generic.List<TextureBank> texBanks;
        private IndexBuffer iBuffer;
        private float texelSize;

        private Effect effect;
        private EffectHandle m1h;
        private EffectHandle t1h;
        private EffectHandle t2h;
        private EffectHandle t3h;
        private EffectHandle t4h;

        public CellTexCreator(int Res) {
            texBanks = new System.Collections.Generic.List<TextureBank>();
            texelSize = 1.0f / (float)Res;

            // Create basic vertex buffer that can be used for all cells which has positions and texture coordinates
            vBuffer = new VertexBuffer(DXMain.device, CellVertex.Stride * 65 * 65, Usage.WriteOnly, CellVertex.Format, Pool.Managed);
            DataStream CellData = vBuffer.Lock(0, 0, LockFlags.None);

            for (int y = 0; y <= 64; y++) {
                for (int x = 0; x <= 64; x++) {
                    CellVertex cv;

                    // Vertex position
                    cv.x = ((float)x / 64.0f) * 2.0f - 1.0f;
                    cv.y = ((float)y / 64.0f) * 2.0f - 1.0f;
                    cv.z = 0.5f;
                    cv.w = 1.0f;
                    // Textures repeat 16 times across a cell
                    cv.u = (float)x / 4.0f;
                    cv.v = (float)y / 4.0f;

                    CellData.Write(cv);
                }
            }
            vBuffer.Unlock();

            // Create triangle strip index buffer
            // Size is 2r + 2rc + 2(r-1) where r is rows and c is colums (squares, not vertices)
            iBuffer = new IndexBuffer(DXMain.device, sizeof(Int16) * 8446, Usage.WriteOnly, Pool.Managed, true);
            DataStream iBuf = iBuffer.Lock(0, 0, LockFlags.None);
            for (int y = 0; y < 64; y++) {
                // If this is is a continuation strip, we need to add two extra vertices to create degenerat triangles
                // and get us back to the left side
                if (y > 0) {
                    iBuf.Write((Int16)(y * 65 + (63 + 1)));
                    iBuf.Write((Int16)(y * 65 + 0));
                }

                // Start the row off with a vertex in the lower left corner of the square
                iBuf.Write((Int16)(y * 65 + 0));

                for (int x = 0; x < 64; x++) {
                    // Add the top left and bottom right vertex of each square
                    iBuf.Write((Int16)((y + 1) * 65 + x));
                    iBuf.Write((Int16)(y * 65 + (x + 1)));
                }

                // End the row with the top right vertex
                iBuf.Write((Int16)((y + 1) * 65 + (63 + 1)));
            }

            iBuffer.Unlock();

            // Create the buffers that will contain different information during each render
            colorBuffer = new VertexBuffer(DXMain.device, NormalColorVertex.Stride * 65 * 65, Usage.WriteOnly, NormalColorVertex.Format, Pool.Managed);

            ResetColorsAndNormals();

            effect = Effect.FromFile(DXMain.device, EffectPath, ShaderFlags.None);

            m1h = effect.GetParameter(null, "transform");
            t1h = effect.GetParameter(null, "t1");
            t2h = effect.GetParameter(null, "t2");
            t3h = effect.GetParameter(null, "t3");
            t4h = effect.GetParameter(null, "t4");

        }

        public void ResetColorsAndNormals() {
            // By default, the normal will be up and the color will be white
            DataStream ColorNormalData = colorBuffer.Lock(0, 0, LockFlags.None);
            NormalColorVertex defaultncv;
            defaultncv.r = 255;
            defaultncv.g = 255;
            defaultncv.b = 255;
            defaultncv.a = 255;

            defaultncv.nx = 0.0f;
            defaultncv.ny = 0.0f;
            defaultncv.nz = 1.0f;

            for (int y = 0; y <= 64; y++) {
                for (int x = 0; x <= 64; x++) {
                    ColorNormalData.Write(defaultncv);
                }
            }
            colorBuffer.Unlock();
        }

        public void SetDefaultCell(LTEX tex) {
            ResetColorsAndNormals();
            texBanks.Clear();
            var tb = new TextureBank();
            tb.SetSingleTexture(tex);
            texBanks.Add(tb);
        }

        public void SetCell(LAND cell) {
            // Write the new colors and normals into the color buffer
            DataStream ColorNormalData = colorBuffer.Lock(0, 0, LockFlags.None);
            NormalColorVertex ncv;

            for (int y = 0; y <= 64; y++) {
                for (int x = 0; x <= 64; x++) {
                    ncv.r = cell.Color[x, y].r;
                    ncv.g = cell.Color[x, y].g;
                    ncv.b = cell.Color[x, y].b;
                    ncv.a = 255;

                    ncv.nx = cell.Normals[x, y].X;
                    ncv.ny = cell.Normals[x, y].Y;
                    ncv.nz = cell.Normals[x, y].Z;

                    ColorNormalData.Write(ncv);
                }
            }
            colorBuffer.Unlock();

            // Dispose of any current texture banks
            texBanks.Clear();

            // Group the unique textures in this cell in fours

            // Find all the unique textures in this cell, match Morrowind rounding
            var tex_dict = new System.Collections.Generic.Dictionary<string, LTEX>();
            for (int y = 0; y <= 64; ++y) {
                for (int x = 0; x <= 64; ++x) {
                    int cell_x = cell.xpos;
                    int cell_y = cell.ypos;
                    int tex_x = (int)Math.Floor(((float)x - 2.0f) / 4.0f);
                    int tex_y = (int)Math.Ceiling(((float)y - 2.0f) / 4.0f);

                    DistantLandForm.ModCell(ref cell_x, ref tex_x);
                    DistantLandForm.ModCell(ref cell_y, ref tex_y);

                    LTEX tmp = DistantLandForm.GetTex(cell_x, cell_y, tex_x, tex_y);
                    string idx = tmp.FilePath;
                    tex_dict[idx] = tmp;
                }
            }

            // Create one bank for each group of 4 textures
            int index = 0;
            var tb = new TextureBank();
            foreach (LTEX tex in tex_dict.Values) {
                switch (index) {
                    case 0:
                        tb.t1 = tex;
                        ++index;
                        break;
                    case 1:
                        tb.t2 = tex;
                        ++index;
                        break;
                    case 2:
                        tb.t3 = tex;
                        ++index;
                        break;
                    case 3:
                        tb.t4 = tex;
                        texBanks.Add(tb);
                        tb = new TextureBank();
                        index = 0;
                        break;
                }
            }

            if (index != 0) {
                texBanks.Add(tb);
            }

            // Calculate weights for all banks
            foreach (TextureBank bank in texBanks) {
                bank.CalcWeights(cell);
            }
        }

        public void Dispose() {
            vBuffer.Dispose();
            iBuffer.Dispose();
            colorBuffer.Dispose();
            texBanks.Clear();
            effect.Dispose();
        }

        public void Begin() {
            DXMain.device.SetRenderState(RenderState.CullMode, Cull.Counterclockwise);
            DXMain.device.SetRenderState(RenderState.Clipping, true);
            DXMain.device.VertexFormat = CellVertex.Format;

            DXMain.device.SetStreamSource(0, vBuffer, 0, CellVertex.Stride);
            DXMain.device.SetStreamSource(1, colorBuffer, 0, NormalColorVertex.Stride);
            var decl = new VertexDeclaration(DXMain.device, Elements);
            DXMain.device.Indices = iBuffer;
            DXMain.device.VertexDeclaration = decl;
        }

        public void BeginNormalMap() {
            DXMain.device.SetRenderState(RenderState.CullMode, Cull.Counterclockwise);
            DXMain.device.SetRenderState(RenderState.Clipping, true);
            DXMain.device.VertexFormat = CellVertex.Format;

            DXMain.device.SetStreamSource(0, vBuffer, 0, CellVertex.Stride);
            DXMain.device.SetStreamSource(1, colorBuffer, 0, NormalColorVertex.Stride);
            var decl = new VertexDeclaration(DXMain.device, NormalElements);
            DXMain.device.Indices = iBuffer;
            DXMain.device.VertexDeclaration = decl;
        }

        public void Render(float pos_x, float pos_y, float scale_x, float scale_y) {
            // Modelview matrix corrects D3D9 half-texel offset (*2 here, as NDC space is from -1 to +1)
            SlimDX.Matrix mat = SlimDX.Matrix.Identity;
            mat.M41 = pos_x - texelSize;
            mat.M42 = pos_y + texelSize;
            mat.M11 = scale_x;
            mat.M22 = scale_y;

            effect.SetValue(m1h, mat);

            foreach (TextureBank bank in texBanks) {
                effect.SetTexture(t1h, bank.t1.tex);
                if (bank.t2 != null) {
                    effect.SetTexture(t2h, bank.t2.tex);
                } else {
                    effect.SetTexture(t2h, bank.t1.tex);
                }
                if (bank.t3 != null) {
                    effect.SetTexture(t3h, bank.t3.tex);
                } else {
                    effect.SetTexture(t3h, bank.t1.tex);
                }
                if (bank.t4 != null) {
                    effect.SetTexture(t4h, bank.t4.tex);
                } else {
                    effect.SetTexture(t4h, bank.t1.tex);
                }

                effect.CommitChanges();
                DXMain.device.SetStreamSource(2, bank.wBuffer, 0, TextureBank.WeightVertex.Stride);
                DXMain.device.BeginScene();
                effect.Begin(FX.None);
                effect.BeginPass(0);
                DXMain.device.DrawIndexedPrimitives(PrimitiveType.TriangleStrip, 0, 0, 4225, 0, 8444);
                effect.EndPass();
                effect.End();
                DXMain.device.EndScene();
            }

            DXMain.device.BeginScene();
            effect.Begin(FX.None);
            effect.BeginPass(2);
            DXMain.device.DrawIndexedPrimitives(PrimitiveType.TriangleStrip, 0, 0, 4225, 0, 8444);
            effect.EndPass();
            effect.End();
            DXMain.device.EndScene();
        }

        public void RenderNormalMap(float pos_x, float pos_y, float scale_x, float scale_y) {
            // Modelview matrix corrects D3D9 half-texel offset (*2 here, as NDC space is from -1 to +1)
            SlimDX.Matrix mat = SlimDX.Matrix.Identity;
            mat.M41 = pos_x - texelSize;
            mat.M42 = pos_y + texelSize;
            mat.M11 = scale_x;
            mat.M22 = scale_y;

            effect.SetValue(m1h, mat);

            effect.CommitChanges();
            DXMain.device.BeginScene();
            effect.Begin(FX.None);
            effect.BeginPass(1);
            DXMain.device.DrawIndexedPrimitives(PrimitiveType.TriangleStrip, 0, 0, 4225, 0, 8444);
            effect.EndPass();
            effect.End();
            DXMain.device.EndScene();
        }

        public void End() {
        }

        public void EndNormalMap() {
            End();
        }
    }

    class WorldTexCreator {
        private const string DefaultTex = @"data files\distantland\default.dds";

        private Texture CompressedTex;
        private Texture UncompressedTex;
        private Texture RenderTargetTex;
        private Surface RenderTarget;

        private int MapSpanX, MapSpanY;
        public float x_scale, y_scale, x_spacing, y_spacing;

        public WorldTexCreator(int Res, int map_span_x, int map_span_y) {
            RenderTargetTex = new Texture(DXMain.device, Res, Res, 0, Usage.RenderTarget, Format.X8R8G8B8, Pool.Default);
            CompressedTex = new Texture(DXMain.device, Res, Res, 0, Usage.None, Format.Dxt1, Pool.SystemMemory);
            UncompressedTex = new Texture(DXMain.device, Res, Res, 0, Usage.None, Format.X8R8G8B8, Pool.SystemMemory);
            RenderTarget = RenderTargetTex.GetSurfaceLevel(0);

            MapSpanX = map_span_x;
            MapSpanY = map_span_y;

            x_scale = 1.0f / (float)MapSpanX;
            y_scale = 1.0f / (float)MapSpanY;
            x_spacing = x_scale * 2.0f;
            y_spacing = y_scale * 2.0f;
        }

        public void Begin() {
            Surface rt = DXMain.device.GetRenderTarget(0);
            if (rt != RenderTarget) {
                DXMain.device.SetRenderTarget(0, RenderTarget);
            }
            rt.Dispose();

            DXMain.device.Clear(ClearFlags.Target, 0, 0.0f, 0);
        }

        public void FinishCompressed(string path, bool isSRGB) {
            Surface tmp = UncompressedTex.GetSurfaceLevel(0);
            Surface.FromSurface(tmp, RenderTarget, Filter.None, 0);
            tmp.Dispose();

            // Generate mips
            Filter filter = Filter.Triangle | (isSRGB ? Filter.Srgb : 0);
            UncompressedTex.FilterTexture(0, filter);

            // Compress mips
            for (int i = 0; i < CompressedTex.LevelCount; i++) {
                Surface dest = CompressedTex.GetSurfaceLevel(i);
                Surface src = UncompressedTex.GetSurfaceLevel(i);
                Surface.FromSurface(dest, src, Filter.None, 0);
            }

            Texture.ToFile(CompressedTex, path, ImageFileFormat.Dds);
        }

        public void FinishUncompressed(string path, bool isSRGB) {
            Surface tmp = UncompressedTex.GetSurfaceLevel(0);
            Surface.FromSurface(tmp, RenderTarget, Filter.None, 0);
            tmp.Dispose();

            // Generate mips
            Filter filter = Filter.Triangle | (isSRGB ? Filter.Srgb : 0);
            UncompressedTex.FilterTexture(0, filter);

            Texture.ToFile(UncompressedTex, path, ImageFileFormat.Dds);
        }

        public void Dispose() {
            RenderTarget.Dispose();
            CompressedTex.Dispose();
            UncompressedTex.Dispose();
            RenderTargetTex.Dispose();
        }
    };
}
