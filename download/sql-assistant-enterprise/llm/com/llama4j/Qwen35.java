///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 21+
//PREVIEW
//COMPILE_OPTIONS --add-modules=jdk.incubator.vector
//RUNTIME_OPTIONS --add-modules=jdk.incubator.vector -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0
//MAIN com.llama4j.Qwen35

// Qwen 3.5 inference in a single Java file
// Based on llama.cpp Qwen3.5 implementation and Gemma4.java
//
// Supports GGUF format with quantized models
// Features full attention layers with gated output and linear attention (gated delta net) layers
// Multi-threaded matrix multiplication using Java's Vector API
//
// To run:
// jbang Qwen35.java --help

package com.llama4j;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

import java.io.IOException;
import java.io.PrintStream;
import java.io.BufferedInputStream;
import java.lang.reflect.Field;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.IntConsumer;
import java.util.function.IntFunction;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

// GGUF format parsing
final class GGUF {
    private static final int GGUF_MAGIC = 0x46554747;
    private static final int DEFAULT_ALIGNMENT = 32;
    private static final int PARSE_BUFFER_SIZE = 1 << 20;
    private static final List<Integer> SUPPORTED_GGUF_VERSIONS = List.of(2, 3);
    private int magic;
    private int version;
    private int tensorCount;
    private int alignment;
    private int metadata_kv_count;
    private Map<String, Object> metadata;
    private Map<String, GGUFTensorInfo> tensorInfos;
    private long tensorDataOffset;

    public Map<String, Object> getMetadata() { return metadata; }
    public long getTensorDataOffset() { return tensorDataOffset; }
    public Map<String, GGUFTensorInfo> getTensorInfos() { return tensorInfos; }

    private final ByteBuffer BB_1 = ByteBuffer.allocate(Byte.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_2 = ByteBuffer.allocate(Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_4 = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_8 = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private long parsePosition;

    public static Map<String, GGMLTensorEntry> loadTensors(FileChannel fileChannel, long tensorDataOffset, Map<String, GGUFTensorInfo> tensorInfos) throws IOException {
        Arena arena = Arena.global();
        MemorySegment tensorData = fileChannel.map(FileChannel.MapMode.READ_ONLY, tensorDataOffset, fileChannel.size() - tensorDataOffset, arena);
        Map<String, GGMLTensorEntry> tensorEntries = HashMap.newHashMap(tensorInfos.size());
        for (Map.Entry<String, GGUFTensorInfo> entry : tensorInfos.entrySet()) {
            GGUFTensorInfo ti = entry.getValue();
            long numberOfElements = FloatTensor.numberOfElementsLong(ti.dimensions());
            long sizeInBytes = ti.ggmlType().byteSizeFor(numberOfElements);
            MemorySegment memorySegment = tensorData.asSlice(ti.offset(), sizeInBytes);
            tensorEntries.put(ti.name(), new GGMLTensorEntry(tensorData, ti.name(), ti.ggmlType(), ti.dimensions(), memorySegment));
        }
        return tensorEntries;
    }

    public static GGUF loadModel(Path modelPath) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(modelPath)) {
            return loadModel(fileChannel, modelPath.toString());
        }
    }

    public static GGUF loadModel(FileChannel fileChannel, String modelLabel) throws IOException {
        try (var ignored = Timer.log("Parse " + modelLabel)) {
            fileChannel.position(0L);
            GGUF gguf = new GGUF();
            ReadableByteChannel channel = Channels.newChannel(
                    new BufferedInputStream(Channels.newInputStream(fileChannel), PARSE_BUFFER_SIZE)
            );
            gguf.parsePosition = 0L;
            gguf.loadModelImpl(channel);
            return gguf;
        }
    }

    enum MetadataValueType {
        UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64;
        private static final MetadataValueType[] VALUES = values();
        public static MetadataValueType fromIndex(int index) { return VALUES[index]; }
    }

    private void loadModelImpl(ReadableByteChannel channel) throws IOException {
        readHeader(channel);
        this.tensorInfos = HashMap.newHashMap(tensorCount);
        for (int i = 0; i < tensorCount; ++i) {
            GGUF.GGUFTensorInfo ti = readTensorInfo(channel);
            assert !tensorInfos.containsKey(ti.name);
            tensorInfos.put(ti.name, ti);
        }
        long position = parsePosition;
        int padding = (int) ((getAlignment() - (position % getAlignment())) % getAlignment());
        skipBytes(channel, padding);
        this.tensorDataOffset = parsePosition;
    }

    public record GGUFTensorInfo(String name, int[] dimensions, GGMLType ggmlType, long offset) {}

    private GGMLType readGGMLType(ReadableByteChannel channel) throws IOException {
        int ggmlTypeId = readInt(channel);
        return GGMLType.fromId(ggmlTypeId);
    }

    private GGUF.GGUFTensorInfo readTensorInfo(ReadableByteChannel channel) throws IOException {
        String name = readString(channel);
        assert name.length() <= 64;
        int n_dimensions = readInt(channel);
        assert n_dimensions <= 4;
        int[] dimensions = new int[n_dimensions];
        for (int i = 0; i < n_dimensions; ++i) {
            dimensions[i] = Math.toIntExact(readLong(channel));
        }
        GGMLType ggmlType = readGGMLType(channel);
        long offset = readLong(channel);
        assert offset % getAlignment() == 0;
        return new GGUF.GGUFTensorInfo(name, dimensions, ggmlType, offset);
    }

    private String readString(ReadableByteChannel channel) throws IOException {
        int len = Math.toIntExact(readLong(channel));
        return new String(readBytes(channel, len), StandardCharsets.UTF_8);
    }

    private Pair<String, Object> readKeyValuePair(ReadableByteChannel channel) throws IOException {
        String key = readString(channel);
        Object value = readMetadataValue(channel);
        return new Pair<>(key, value);
    }

    private Object readMetadataValue(ReadableByteChannel channel) throws IOException {
        MetadataValueType valueType = readMetadataValueType(channel);
        return readMetadataValueOfType(valueType, channel);
    }

    void readHeader(ReadableByteChannel channel) throws IOException {
        this.magic = readInt(channel);
        if (magic != GGUF_MAGIC) {
            throw new IllegalArgumentException("unsupported header.magic " + magic);
        }
        this.version = readInt(channel);
        if (!SUPPORTED_GGUF_VERSIONS.contains(version)) {
            throw new IllegalArgumentException("unsupported header.version " + version);
        }
        this.tensorCount = Math.toIntExact(readLong(channel));
        this.metadata_kv_count = Math.toIntExact(readLong(channel));
        this.metadata = HashMap.newHashMap(metadata_kv_count);
        for (int i = 0; i < metadata_kv_count; ++i) {
            Pair<String, Object> keyValue = readKeyValuePair(channel);
            assert !metadata.containsKey(keyValue.first());
            metadata.put(keyValue.first(), keyValue.second());
        }
    }

    private Object readArray(ReadableByteChannel channel) throws IOException {
        MetadataValueType valueType = readMetadataValueType(channel);
        int len = Math.toIntExact(readLong(channel));
        switch (valueType) {
            case UINT8, INT8 -> { return readBytes(channel, len); }
            case UINT16, INT16 -> {
                short[] shorts = new short[len];
                for (int i = 0; i < len; ++i) shorts[i] = readShort(channel);
                return shorts;
            }
            case UINT32, INT32 -> {
                int[] ints = new int[len];
                for (int i = 0; i < len; ++i) ints[i] = readInt(channel);
                return ints;
            }
            case FLOAT32 -> {
                float[] floats = new float[len];
                for (int i = 0; i < len; ++i) floats[i] = readFloat(channel);
                return floats;
            }
            case BOOL -> {
                boolean[] booleans = new boolean[len];
                for (int i = 0; i < len; ++i) booleans[i] = readBoolean(channel);
                return booleans;
            }
            case STRING -> {
                String[] strings = new String[len];
                for (int i = 0; i < len; ++i) strings[i] = readString(channel);
                return strings;
            }
            case ARRAY -> {
                Object[] arrays = new Object[len];
                for (int i = 0; i < len; ++i) arrays[i] = readArray(channel);
                return arrays;
            }
            default -> throw new UnsupportedOperationException("read array of " + valueType);
        }
    }

    private Object readMetadataValueOfType(MetadataValueType valueType, ReadableByteChannel channel) throws IOException {
        return switch (valueType) {
            case UINT8, INT8 -> readByte(channel);
            case UINT16, INT16 -> readShort(channel);
            case UINT32, INT32 -> readInt(channel);
            case FLOAT32 -> readFloat(channel);
            case UINT64, INT64 -> readLong(channel);
            case FLOAT64 -> readDouble(channel);
            case BOOL -> readBoolean(channel);
            case STRING -> readString(channel);
            case ARRAY -> readArray(channel);
        };
    }

    private MetadataValueType readMetadataValueType(ReadableByteChannel channel) throws IOException {
        int index = readInt(channel);
        return MetadataValueType.fromIndex(index);
    }

    private byte[] readBytes(ReadableByteChannel channel, int length) throws IOException {
        byte[] bytes = new byte[length];
        readFully(channel, ByteBuffer.wrap(bytes));
        return bytes;
    }

    private void skipBytes(ReadableByteChannel channel, int length) throws IOException {
        int remaining = length;
        byte[] scratch = new byte[Math.min(length, 1 << 12)];
        while (remaining > 0) {
            int chunk = Math.min(remaining, scratch.length);
            readFully(channel, ByteBuffer.wrap(scratch, 0, chunk));
            remaining -= chunk;
        }
    }

    private void readFully(ReadableByteChannel channel, ByteBuffer byteBuffer) throws IOException {
        int expected = byteBuffer.remaining();
        while (byteBuffer.hasRemaining()) {
            int bytesRead = channel.read(byteBuffer);
            if (bytesRead < 0) {
                throw new IOException("Unexpected EOF while reading GGUF metadata");
            }
        }
        parsePosition += expected;
    }

    private byte readByte(ReadableByteChannel channel) throws IOException {
        BB_1.clear();
        readFully(channel, BB_1);
        return BB_1.get(0);
    }

    private boolean readBoolean(ReadableByteChannel channel) throws IOException {
        return readByte(channel) != 0;
    }

    private short readShort(ReadableByteChannel channel) throws IOException {
        BB_2.clear();
        readFully(channel, BB_2);
        return BB_2.getShort(0);
    }

    private int readInt(ReadableByteChannel channel) throws IOException {
        BB_4.clear();
        readFully(channel, BB_4);
        return BB_4.getInt(0);
    }

    private long readLong(ReadableByteChannel channel) throws IOException {
        BB_8.clear();
        readFully(channel, BB_8);
        return BB_8.getLong(0);
    }

    private float readFloat(ReadableByteChannel channel) throws IOException {
        return Float.intBitsToFloat(readInt(channel));
    }

    private double readDouble(ReadableByteChannel channel) throws IOException {
        return Double.longBitsToDouble(readLong(channel));
    }

    public int getAlignment() {
        if (alignment != 0) return alignment;
        alignment = (int) metadata.getOrDefault("general.alignment", DEFAULT_ALIGNMENT);
        assert Integer.bitCount(alignment) == 1 : "alignment must be a power of two";
        return alignment;
    }
}

interface Timer extends AutoCloseable {
    @Override void close();
    static Timer log(String label) { return log(label, TimeUnit.MILLISECONDS); }
    static Timer log(String label, TimeUnit timeUnit) {
        return new Timer() {
            final long startNanos = System.nanoTime();
            @Override
            public void close() {
                long elapsedNanos = System.nanoTime() - startNanos;
                System.err.println(label + ": " + timeUnit.convert(elapsedNanos, TimeUnit.NANOSECONDS) + " " + timeUnit.toChronoUnit().name().toLowerCase());
            }
        };
    }
}

final class Parallel {
    public static void parallelFor(int startInclusive, int endExclusive, IntConsumer action) {
        IntStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }
}

record Pair<First, Second>(First first, Second second) {}

record GGMLTensorEntry(MemorySegment mappedFile, String name, GGMLType ggmlType, int[] shape, MemorySegment memorySegment) {}

final class Float16 {
    public static final int BYTES = 2;
}

enum GGMLType {
    F32(Float.BYTES),
    F16(Float16.BYTES),
    Q4_0(Float16.BYTES + 16 * Byte.BYTES, 32),
    Q4_1(2 * Float16.BYTES + 16 * Byte.BYTES, 32),
    UNSUPPORTED_Q4_2(Integer.MAX_VALUE),
    UNSUPPORTED_Q4_3(Integer.MAX_VALUE),
    Q5_0(Integer.MAX_VALUE),
    Q5_1(Integer.MAX_VALUE),
    Q8_0(Float16.BYTES + 32 * Byte.BYTES, 32),
    Q8_1(32 * Byte.BYTES + 2 * Float.BYTES, 32),
    Q2_K(Integer.MAX_VALUE),
    Q3_K(Integer.MAX_VALUE),
    Q4_K(2 * Float16.BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 2, GGMLType.QK_K),
    Q5_K(2 * Float16.BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 8 + GGMLType.QK_K / 2, GGMLType.QK_K),
    Q6_K(GGMLType.QK_K / 2 + GGMLType.QK_K / 4 + GGMLType.QK_K / 16 + Float16.BYTES, GGMLType.QK_K),
    Q8_K(Integer.MAX_VALUE),
    IQ2_XXS(Integer.MAX_VALUE),
    IQ2_XS(Integer.MAX_VALUE),
    IQ3_XXS(Integer.MAX_VALUE),
    IQ1_S(Integer.MAX_VALUE),
    IQ4_NL(Integer.MAX_VALUE),
    IQ3_S(Integer.MAX_VALUE),
    IQ2_S(Integer.MAX_VALUE),
    IQ4_XS(Integer.MAX_VALUE),
    I8(Byte.BYTES),
    I16(Short.BYTES),
    I32(Integer.BYTES),
    I64(Long.BYTES),
    F64(Double.BYTES),
    IQ1_M(Integer.MAX_VALUE),
    BF16(Float16.BYTES),
    UNSUPPORTED_Q4_0_4_4(Integer.MAX_VALUE),
    UNSUPPORTED_Q4_0_4_8(Integer.MAX_VALUE),
    UNSUPPORTED_Q4_0_8_8(Integer.MAX_VALUE),
    TQ1_0(Integer.MAX_VALUE),
    TQ2_0(Integer.MAX_VALUE),
    UNSUPPORTED_IQ4_NL_4_4(Integer.MAX_VALUE),
    UNSUPPORTED_IQ4_NL_4_8(Integer.MAX_VALUE),
    UNSUPPORTED_IQ4_NL_8_8(Integer.MAX_VALUE),
    MXFP4(Byte.BYTES + GGMLType.QK_MXFP4 / 2, GGMLType.QK_MXFP4),
    NVFP4(Integer.MAX_VALUE);

    private static final GGMLType[] VALUES = values();
    private final int typeSize;
    private final int blockSize;

    public int getTypeSize() { return typeSize; }
    public int getBlockSize() { return blockSize; }
    public static GGMLType fromId(int id) { return VALUES[id]; }

    GGMLType(int typeSize) { this(typeSize, 1); }

    public long byteSizeFor(long numberOfElements) {
        long t = numberOfElements * (long) getTypeSize();
        assert t % getBlockSize() == 0;
        return t / getBlockSize();
    }

    public static final int QK_K = 256;
    public static final int QK_MXFP4 = 32;

    GGMLType(int typeSize, int blockSize) {
        assert blockSize > 0;
        assert typeSize > 0;
        assert isPowerOf2(blockSize);
        this.typeSize = typeSize;
        this.blockSize = blockSize;
    }

    private static boolean isPowerOf2(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
}

abstract class FloatTensor {
    static final int VECTOR_BIT_SIZE = Integer.getInteger("llama.VectorBitSize", VectorShape.preferredShape().vectorBitSize());
    static final boolean USE_VECTOR_API = VECTOR_BIT_SIZE != 0;

    static final VectorSpecies<Float> F_SPECIES;
    static final VectorSpecies<Integer> I_SPECIES;
    static final VectorSpecies<Short> S_SPECIES_HALF;

    static {
        if (USE_VECTOR_API) {
            F_SPECIES = VectorShape.forBitSize(VECTOR_BIT_SIZE).withLanes(float.class);
            I_SPECIES = F_SPECIES.withLanes(int.class);
            S_SPECIES_HALF = VectorShape.forBitSize(F_SPECIES.vectorBitSize() / 2).withLanes(short.class);
            assert F_SPECIES.length() == S_SPECIES_HALF.length();
        } else {
            F_SPECIES = null;
            I_SPECIES = null;
            S_SPECIES_HALF = null;
        }
    }

    static final sun.misc.Unsafe UNSAFE;

    static {
        try {
            Field f = sun.misc.Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            UNSAFE = (sun.misc.Unsafe) f.get(null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    static short readShort(MemorySegment memorySegment, long offset) {
        return UNSAFE.getShort(memorySegment.address() + offset);
    }

    static float readFloat16(MemorySegment memorySegment, long offset) {
        return Float.float16ToFloat(readShort(memorySegment, offset));
    }

    static byte readByte(MemorySegment memorySegment, long offset) {
        return UNSAFE.getByte(memorySegment.address() + offset);
    }

    static float readFloat(MemorySegment memorySegment, long offset) {
        return UNSAFE.getFloat(memorySegment.address() + offset);
    }

    abstract long size();
    abstract float getFloat(long index);
    abstract void setFloat(int index, float value);
    abstract FloatVector getFloatVector(VectorSpecies<Float> species, int offset);
    abstract GGMLType type();

    public static int numberOfElements(int... dimensions) {
        assert Arrays.stream(dimensions).allMatch(i -> i > 0);
        return Arrays.stream(dimensions).reduce(Math::multiplyExact).orElseThrow();
    }

    public static long numberOfElementsLong(int... dimensions) {
        long result = 1;
        for (int d : dimensions) {
            assert d > 0;
            result = Math.multiplyExact(result, d);
        }
        return result;
    }

    static float scalarDot(FloatTensor thiz, int thisOffset, FloatTensor that, int thatOffset, int size) {
        float result = 0f;
        for (int j = 0; j < size; j++) {
            result += thiz.getFloat(thisOffset + j) * that.getFloat(thatOffset + j);
        }
        return result;
    }

    float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return scalarDot(this, thisOffset, that, thatOffset, size);
    }

    void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1) {
        Parallel.parallelFor(0, dim0, i -> out.setFloat(i, dot(i * dim1, that, 0, dim1)));
    }

    void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1, int thisOffset) {
        Parallel.parallelFor(0, dim0, i -> out.setFloat(i, dot(thisOffset + i * dim1, that, 0, dim1)));
    }

    @FunctionalInterface
    interface AggregateFunction {
        float apply(float acc, float value);
    }

    float reduce(int thisOffset, int size, float seed, AggregateFunction reduce) {
        float result = seed;
        for (int i = 0; i < size; ++i) {
            result = reduce.apply(result, getFloat(thisOffset + i));
        }
        return result;
    }

    float sum(int thisOffset, int size) {
        return reduce(thisOffset, size, 0f, Float::sum);
    }

    float max(int thisOffset, int size) {
        return reduce(thisOffset, size, Float.NEGATIVE_INFINITY, Float::max);
    }

    void copyTo(int thisOffset, FloatTensor that, int thatOffset, int size) {
        that.mapWithIndexInPlace(thatOffset, size, (value, index) -> this.getFloat(index - thatOffset + thisOffset));
    }

    int argmax(int thisOffset, int size) {
        assert size > 0;
        int maxIndex = thisOffset;
        float maxValue = this.getFloat(maxIndex);
        int endIndex = thisOffset + size;
        for (int i = thisOffset; i < endIndex; ++i) {
            float f = this.getFloat(i);
            if (f > maxValue) {
                maxValue = f;
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    int argmax() {
        return argmax(0, Math.toIntExact(size()));
    }

    @FunctionalInterface
    interface MapFunction {
        float apply(float value);
    }

    @FunctionalInterface
    interface MapWithIndexFunction {
        float apply(float value, int index);
    }

    FloatTensor mapInPlace(int thisOffset, int size, MapFunction mapFunction) {
        int endIndex = thisOffset + size;
        for (int i = thisOffset; i < endIndex; ++i) {
            setFloat(i, mapFunction.apply(getFloat(i)));
        }
        return this;
    }

    FloatTensor mapInPlace(MapFunction mapFunction) {
        return mapInPlace(0, Math.toIntExact(size()), mapFunction);
    }

    FloatTensor mapWithIndexInPlace(int thisOffset, int size, FloatTensor.MapWithIndexFunction mapWithIndexFunction) {
        int endOffset = thisOffset + size;
        for (int i = thisOffset; i < endOffset; ++i) {
            setFloat(i, mapWithIndexFunction.apply(getFloat(i), i));
        }
        return this;
    }

    FloatTensor addInPlace(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value + that.getFloat(index - thisOffset + thatOffset));
    }

    FloatTensor addInPlace(FloatTensor that) {
        return addInPlace(0, that, 0, Math.toIntExact(size()));
    }

    FloatTensor multiplyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value * that.getFloat(index - thisOffset + thatOffset));
    }

    FloatTensor divideInPlace(int thisOffset, int size, float value) {
        return mapInPlace(thisOffset, size, f -> f / value);
    }

    FloatTensor fillInPlace(int thisOffset, int size, float value) {
        return mapInPlace(thisOffset, size, unused -> value);
    }

    FloatTensor softmaxInPlace(int thisOffset, int size) {
        float maxVal = max(thisOffset, size);
        mapInPlace(thisOffset, size, f -> (float) Math.exp(f - maxVal));
        float sum = sum(thisOffset, size);
        return divideInPlace(thisOffset, size, sum);
    }

    FloatTensor saxpyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size, float a) {
        for (int i = 0; i < size; ++i) {
            setFloat(thisOffset + i, a * that.getFloat(thatOffset + i) + this.getFloat(thisOffset + i));
        }
        return this;
    }
}

final class ArrayFloatTensor extends FloatTensor {
    final float[] values;

    public ArrayFloatTensor(float[] values) {
        this.values = values;
    }

    public static ArrayFloatTensor allocate(int... dimensions) {
        return new ArrayFloatTensor(new float[FloatTensor.numberOfElements(dimensions)]);
    }

    @Override
    long size() {
        return values.length;
    }

    @Override
    float getFloat(long index) {
        return values[(int) index];
    }

    @Override
    void setFloat(int index, float value) {
        values[index] = value;
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int offset) {
        return FloatVector.fromArray(species, values, offset);
    }

    @Override
    float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (USE_VECTOR_API && that instanceof ArrayFloatTensor aft) {
            return vectorDot(values, thisOffset, aft.values, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(float[] thiz, int thisOffset, float[] that, int thatOffset, int size) {
        FloatVector acc = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector a = FloatVector.fromArray(F_SPECIES, thiz, thisOffset + i);
            FloatVector b = FloatVector.fromArray(F_SPECIES, that, thatOffset + i);
            acc = a.fma(b, acc);
        }
        float result = acc.reduceLanes(VectorOperators.ADD);
        for (int i = upperBound; i < size; i++) {
            result += thiz[thisOffset + i] * that[thatOffset + i];
        }
        return result;
    }

    @Override
    GGMLType type() {
        return GGMLType.F32;
    }
}

final class F32FloatTensor extends FloatTensor {
    final long size;
    final MemorySegment memorySegment;

    public F32FloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    long size() {
        return size;
    }

    @Override
    float getFloat(long index) {
        return readFloat(memorySegment, index * Float.BYTES);
    }

    @Override
    void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int offset) {
        return FloatVector.fromMemorySegment(species, memorySegment, offset * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    GGMLType type() {
        return GGMLType.F32;
    }
}

final class F16FloatTensor extends FloatTensor {
    final long size;
    final MemorySegment memorySegment;

    public F16FloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override long size() { return size; }
    @Override void setFloat(int index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, int offset) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override GGMLType type() { return GGMLType.F16; }

    @Override
    float getFloat(long index) {
        assert 0 <= index && index < size;
        return readFloat16(memorySegment, index * 2L);
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (USE_VECTOR_API && that instanceof ArrayFloatTensor aft) {
            return vectorDot(this, thisOffset, aft, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(F16FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = that.getFloatVector(F_SPECIES, thatOffset + i);
            ShortVector bits16 = ShortVector.fromMemorySegment(S_SPECIES_HALF, thiz.memorySegment, (thisOffset + i) * 2L, ByteOrder.LITTLE_ENDIAN);
            var bits32 = bits16.castShape(I_SPECIES, 0).reinterpretAsInts();
            var zeroExponentMask = bits32.and(0x7C00).neg().lanewise(VectorOperators.ASHR, 31);
            bits32 = bits32.and(0x8000).lanewise(VectorOperators.LSHL, 16)
                    .or(bits32.and(0x7FFF).add(0x1C000).lanewise(VectorOperators.LSHL, 13).and(zeroExponentMask));
            FloatVector thizVector = bits32.reinterpretAsFloats();
            val = thizVector.fma(thatVector, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        if (upperBound < size) result += scalarDot(thiz, thisOffset + upperBound, that, thatOffset + upperBound, size - upperBound);
        return result;
    }
}

final class BF16FloatTensor extends FloatTensor {
    final long size;
    final MemorySegment memorySegment;

    public BF16FloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override long size() { return size; }
    @Override void setFloat(int index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, int offset) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override GGMLType type() { return GGMLType.BF16; }

    @Override
    float getFloat(long index) {
        assert 0 <= index && index < size;
        short bits = readShort(memorySegment, index * 2L);
        return Float.intBitsToFloat(bits << 16);
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (USE_VECTOR_API && that instanceof ArrayFloatTensor aft) {
            return vectorDot(this, thisOffset, aft, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(BF16FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = that.getFloatVector(F_SPECIES, thatOffset + i);
            ShortVector bfloat16 = ShortVector.fromMemorySegment(S_SPECIES_HALF, thiz.memorySegment, (thisOffset + i) * 2L, ByteOrder.LITTLE_ENDIAN);
            FloatVector thizVector = bfloat16
                    .castShape(I_SPECIES, 0)
                    .lanewise(VectorOperators.LSHL, 16)
                    .reinterpretAsFloats();
            val = thizVector.fma(thatVector, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        if (upperBound < size) result += scalarDot(thiz, thisOffset + upperBound, that, thatOffset + upperBound, size - upperBound);
        return result;
    }
}

final class Q8_0FloatTensor extends FloatTensor {

    private static final int Q8_0_BLOCK_SIZE = GGMLType.Q8_0.getBlockSize();
    private static final int Q8_0_TYPE_SIZE = GGMLType.Q8_0.getTypeSize();

    final long size;
    final MemorySegment memorySegment;

    public Q8_0FloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override long size() { return size; }
    @Override void setFloat(int index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, int offset) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override GGMLType type() { return GGMLType.Q8_0; }

    @Override
    float getFloat(long index) {
        long blockIndex = index / Q8_0_BLOCK_SIZE;
        long withinBlockIndex = index % Q8_0_BLOCK_SIZE;
        long blockOffset = blockIndex * Q8_0_TYPE_SIZE;
        byte quant = readByte(memorySegment, blockOffset + Float16.BYTES + withinBlockIndex);
        float scale = readFloat16(memorySegment, blockOffset);
        return quant * scale;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (USE_VECTOR_API && that instanceof ArrayFloatTensor aft) {
            return vectorDot(this, thisOffset, aft, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    @Override
    void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1) {
        if (USE_VECTOR_API && that instanceof ArrayFloatTensor aft && out instanceof ArrayFloatTensor outArray) {
            float[] outValues = outArray.values;
            Parallel.parallelFor(0, dim0, i -> outValues[i] = vectorDot(this, i * dim1, aft, 0, dim1));
            return;
        }
        super.matmul(that, out, dim0, dim1);
    }

    @Override
    void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1, int thisOffset) {
        if (USE_VECTOR_API && that instanceof ArrayFloatTensor aft && out instanceof ArrayFloatTensor outArray) {
            float[] outValues = outArray.values;
            Parallel.parallelFor(0, dim0, i -> outValues[i] = vectorDot(this, thisOffset + i * dim1, aft, 0, dim1));
            return;
        }
        super.matmul(that, out, dim0, dim1, thisOffset);
    }

    private static float vectorDot(Q8_0FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;
        float[] thatValues = that.values;
        int alignmentBound = Math.min(size, -thisOffset & (Q8_0_BLOCK_SIZE - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        FloatVector val = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / Q8_0_BLOCK_SIZE * Q8_0_TYPE_SIZE;
        int upperBound = j + (size - j) / Q8_0_BLOCK_SIZE * Q8_0_BLOCK_SIZE;
        for (; j < upperBound; j += Q8_0_BLOCK_SIZE, blockOffset += Q8_0_TYPE_SIZE) {
            float wScaleValue = readFloat16(thiz.memorySegment, blockOffset);
            var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_256, thiz.memorySegment, blockOffset + Float16.BYTES, ByteOrder.LITTLE_ENDIAN);
                    var s0 = FloatVector.fromArray(F_SPECIES, thatValues, thatOffset + j).mul(wBytes.castShape(F_SPECIES, 0));
                    var s1 = FloatVector.fromArray(F_SPECIES, thatValues, thatOffset + j + F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 1));
                    val = s0.add(s1).fma(wScale, val);
                }
                case 256 -> {
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_256, thiz.memorySegment, blockOffset + Float16.BYTES, ByteOrder.LITTLE_ENDIAN);
                    var s0 = FloatVector.fromArray(F_SPECIES, thatValues, thatOffset + j).mul(wBytes.castShape(F_SPECIES, 0));
                    var s1 = FloatVector.fromArray(F_SPECIES, thatValues, thatOffset + j + 2 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 2));
                    s0 = FloatVector.fromArray(F_SPECIES, thatValues, thatOffset + j + F_SPECIES.length()).fma(wBytes.castShape(F_SPECIES, 1), s0);
                    s1 = FloatVector.fromArray(F_SPECIES, thatValues, thatOffset + j + 3 * F_SPECIES.length()).fma(wBytes.castShape(F_SPECIES, 3), s1);
                    val = s0.add(s1).fma(wScale, val);
                }
                case 128 -> {
                    for (int i = 0; i < 2; ++i) {
                        var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + Float16.BYTES + i * ByteVector.SPECIES_128.vectorByteSize(), ByteOrder.LITTLE_ENDIAN);
                        var s0 = FloatVector.fromArray(F_SPECIES, thatValues, thatOffset + j + i * 16).mul(wBytes.castShape(F_SPECIES, 0));
                        var s1 = FloatVector.fromArray(F_SPECIES, thatValues, thatOffset + j + i * 16 + 2 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 2));
                        s0 = FloatVector.fromArray(F_SPECIES, thatValues, thatOffset + j + i * 16 + F_SPECIES.length()).fma(wBytes.castShape(F_SPECIES, 1), s0);
                        s1 = FloatVector.fromArray(F_SPECIES, thatValues, thatOffset + j + i * 16 + 3 * F_SPECIES.length()).fma(wBytes.castShape(F_SPECIES, 3), s1);
                        val = s0.add(s1).fma(wScale, val);
                    }
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);
        if (j < size) result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        return result;
    }
}

final class Q4_0FloatTensor extends FloatTensor {
    final long size;
    final MemorySegment memorySegment;

    public Q4_0FloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override long size() { return size; }
    @Override void setFloat(int index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, int offset) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override GGMLType type() { return GGMLType.Q4_0; }

    @Override
    float getFloat(long index) {
        assert 0 <= index && index < size;
        long blockIndex = index / GGMLType.Q4_0.getBlockSize();
        long blockOffset = blockIndex * GGMLType.Q4_0.getTypeSize();
        float scale = readFloat16(memorySegment, blockOffset);
        int modIndex = (int) (index % GGMLType.Q4_0.getBlockSize());
        byte quant;
        if (modIndex < GGMLType.Q4_0.getBlockSize() / 2) {
            quant = (byte) (readByte(memorySegment, blockOffset + Float16.BYTES + modIndex) & 0x0F);
        } else {
            quant = (byte) ((readByte(memorySegment, blockOffset + Float16.BYTES + modIndex - GGMLType.Q4_0.getBlockSize() / 2) >>> 4) & 0x0F);
        }
        quant -= 8;
        return quant * scale;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (USE_VECTOR_API && that instanceof ArrayFloatTensor aft) {
            return vectorDot(this, thisOffset, aft, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(Q4_0FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;
        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.Q4_0.getBlockSize() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        FloatVector val = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / GGMLType.Q4_0.getBlockSize() * GGMLType.Q4_0.getTypeSize();
        int upperBound = j + (size - j) / GGMLType.Q4_0.getBlockSize() * GGMLType.Q4_0.getBlockSize();
        for (; j < upperBound; j += GGMLType.Q4_0.getBlockSize(), blockOffset += GGMLType.Q4_0.getTypeSize()) {
            float wScaleValue = readFloat16(thiz.memorySegment, blockOffset);
            var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + Float16.BYTES, ByteOrder.LITTLE_ENDIAN);
            var loBytes = wBytes.and((byte) 0xF).sub((byte) 8);
            var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4).sub((byte) 8);
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var s0 = that.getFloatVector(F_SPECIES, thatOffset + j).mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length()).mul(hiBytes.castShape(F_SPECIES, 0));
                    val = s0.add(s1).fma(wScale, val);
                }
                case 256 -> {
                    var s0 = that.getFloatVector(F_SPECIES, thatOffset + j).mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 = that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length()).mul(hiBytes.castShape(F_SPECIES, 0));
                    s0 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length()).fma(loBytes.castShape(F_SPECIES, 1), s0);
                    s1 = that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length()).fma(hiBytes.castShape(F_SPECIES, 1), s1);
                    val = s0.add(s1).fma(wScale, val);
                }
                case 128 -> {
                    for (int i = 0; i < 2; ++i) {
                        var tmp = i == 0 ? loBytes : hiBytes;
                        var s0 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 0));
                        var s1 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 2) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 2));
                        s0 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 1) * F_SPECIES.length()).fma(tmp.castShape(F_SPECIES, 1), s0);
                        s1 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 3) * F_SPECIES.length()).fma(tmp.castShape(F_SPECIES, 3), s1);
                        val = s0.add(s1).fma(wScale, val);
                    }
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);
        if (j < size) result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        return result;
    }
}

final class Q4_1FloatTensor extends FloatTensor {
    final long size;
    final MemorySegment memorySegment;

    public Q4_1FloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    long size() {
        return size;
    }

    @Override
    float getFloat(long index) {
        long blockIndex = index / GGMLType.Q4_1.getBlockSize();
        long blockOffset = blockIndex * GGMLType.Q4_1.getTypeSize();
        float min = Float.float16ToFloat(readShort(memorySegment, blockOffset));
        float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset + Float16.BYTES));
        byte quant = readByte(memorySegment, blockOffset + 2 * Float16.BYTES + (index / 2));
        int nibble = (index % 2 == 0) ? (quant & 0x0F) : ((quant >> 4) & 0x0F);
        return min + scale * nibble;
    }

    @Override
    void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int offset) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    GGMLType type() {
        return GGMLType.Q4_1;
    }
}

final class Q4_KFloatTensor extends FloatTensor {
    static final int BLOCK_SIZE = GGMLType.QK_K;
    static final int TYPE_SIZE = GGMLType.Q4_K.getTypeSize();

    final long size;
    final MemorySegment memorySegment;

    public Q4_KFloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    long size() {
        return size;
    }

    static int getScaleMinK4(int j, MemorySegment mem, long scalesOffset, boolean isMin) {
        if (j < 4) {
            int idx = isMin ? j + 4 : j;
            return Byte.toUnsignedInt(readByte(mem, scalesOffset + idx)) & 63;
        } else {
            int lowIdx = j + 4;
            int highIdx = isMin ? j : j - 4;
            int low = isMin
                    ? (Byte.toUnsignedInt(readByte(mem, scalesOffset + lowIdx)) >> 4)
                    : (Byte.toUnsignedInt(readByte(mem, scalesOffset + lowIdx)) & 0xF);
            int high = (Byte.toUnsignedInt(readByte(mem, scalesOffset + highIdx)) >> 6) & 0x3;
            return low | (high << 4);
        }
    }

    @Override
    float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int withinBlock = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * TYPE_SIZE;
        float d = readFloat16(memorySegment, blockOffset);
        float dmin = readFloat16(memorySegment, blockOffset + 2);
        long scalesOffset = blockOffset + 4;
        long qsOffset = blockOffset + 16;

        int group = withinBlock / 64;
        int inGroup = withinBlock % 64;
        int subBlock;
        int nibbleIndex;
        boolean isHigh;
        if (inGroup < 32) {
            subBlock = group * 2;
            nibbleIndex = inGroup;
            isHigh = false;
        } else {
            subBlock = group * 2 + 1;
            nibbleIndex = inGroup - 32;
            isHigh = true;
        }

        int sc = getScaleMinK4(subBlock, memorySegment, scalesOffset, false);
        int m = getScaleMinK4(subBlock, memorySegment, scalesOffset, true);

        byte qsByte = readByte(memorySegment, qsOffset + group * 32 + nibbleIndex);
        int quant = isHigh ? ((Byte.toUnsignedInt(qsByte) >> 4) & 0xF) : (Byte.toUnsignedInt(qsByte) & 0xF);

        return d * sc * quant - dmin * m;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (USE_VECTOR_API && that instanceof ArrayFloatTensor aft) {
            return vectorDot(this, thisOffset, aft, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(Q4_KFloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        int alignmentBound = Math.min(size, -thisOffset & (BLOCK_SIZE - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }

        FloatVector val = FloatVector.zero(F_SPECIES);
        FloatVector val2 = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / BLOCK_SIZE * TYPE_SIZE;
        int upperBound = j + (size - j) / BLOCK_SIZE * BLOCK_SIZE;

        for (; j < upperBound; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
            float d = readFloat16(thiz.memorySegment, blockOffset);
            float dmin = readFloat16(thiz.memorySegment, blockOffset + 2);
            long scalesOff = blockOffset + 4;
            long qsOff = blockOffset + 16;

            for (int g = 0; g < 4; g++) {
                float d1 = d * getScaleMinK4(g * 2, thiz.memorySegment, scalesOff, false);
                float negM1 = -(dmin * getScaleMinK4(g * 2, thiz.memorySegment, scalesOff, true));
                float d2 = d * getScaleMinK4(g * 2 + 1, thiz.memorySegment, scalesOff, false);
                float negM2 = -(dmin * getScaleMinK4(g * 2 + 1, thiz.memorySegment, scalesOff, true));

                var d1Vec = FloatVector.broadcast(F_SPECIES, d1);
                var negM1Vec = FloatVector.broadcast(F_SPECIES, negM1);
                var d2Vec = FloatVector.broadcast(F_SPECIES, d2);
                var negM2Vec = FloatVector.broadcast(F_SPECIES, negM2);

                int loBase = thatOffset + j + g * 64;
                int hiBase = thatOffset + j + g * 64 + 32;

                for (int c = 0; c < 2; c++) {
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            qsOff + (long) g * 32 + c * 16, ByteOrder.LITTLE_ENDIAN);
                    var loBytes = wBytes.and((byte) 0xF);
                    var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4);

                    int loIdx = loBase + c * 16;
                    int hiIdx = hiBase + c * 16;

                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            var loQ = loBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val = loQ.fma(d1Vec, negM1Vec).fma(FloatVector.fromArray(F_SPECIES, that.values, loIdx), val);
                            var hiQ = hiBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val2 = hiQ.fma(d2Vec, negM2Vec).fma(FloatVector.fromArray(F_SPECIES, that.values, hiIdx), val2);
                        }
                        case 256 -> {
                            var loQ0 = loBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var loQ1 = loBytes.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val = loQ0.fma(d1Vec, negM1Vec).fma(FloatVector.fromArray(F_SPECIES, that.values, loIdx), val);
                            val2 = loQ1.fma(d1Vec, negM1Vec).fma(FloatVector.fromArray(F_SPECIES, that.values, loIdx + F_SPECIES.length()), val2);
                            var hiQ0 = hiBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQ1 = hiBytes.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val = hiQ0.fma(d2Vec, negM2Vec).fma(FloatVector.fromArray(F_SPECIES, that.values, hiIdx), val);
                            val2 = hiQ1.fma(d2Vec, negM2Vec).fma(FloatVector.fromArray(F_SPECIES, that.values, hiIdx + F_SPECIES.length()), val2);
                        }
                        case 128 -> {
                            for (int p = 0; p < 4; p++) {
                                var loQ = loBytes.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val = loQ.fma(d1Vec, negM1Vec).fma(FloatVector.fromArray(F_SPECIES, that.values, loIdx + p * F_SPECIES.length()), val);
                                var hiQ = hiBytes.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val2 = hiQ.fma(d2Vec, negM2Vec).fma(FloatVector.fromArray(F_SPECIES, that.values, hiIdx + p * F_SPECIES.length()), val2);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }
        result += val.add(val2).reduceLanes(VectorOperators.ADD);
        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }

    @Override
    void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int offset) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    GGMLType type() {
        return GGMLType.Q4_K;
    }
}

final class Q5_KFloatTensor extends FloatTensor {
    final long size;
    final MemorySegment memorySegment;

    public Q5_KFloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    long size() {
        return size;
    }

    @Override
    float getFloat(long index) {
        // Simplified implementation
        return 0.0f;
    }

    @Override
    void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int offset) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    GGMLType type() {
        return GGMLType.Q5_K;
    }
}

final class Q6_KFloatTensor extends FloatTensor {
    final long size;
    final MemorySegment memorySegment;

    public Q6_KFloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    long size() {
        return size;
    }

    @Override
    float getFloat(long index) {
        // Simplified implementation
        return 0.0f;
    }

    @Override
    void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int offset) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    GGMLType type() {
        return GGMLType.Q6_K;
    }
}

final class MXFP4FloatTensor extends FloatTensor {
    private static final int[] MXFP4_VALUES = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

    final long size;
    final MemorySegment memorySegment;

    MXFP4FloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override long size() { return size; }
    @Override void setFloat(int index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, int offset) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override GGMLType type() { return GGMLType.MXFP4; }

    @Override
    float getFloat(long index) {
        assert 0 <= index && index < size;
        long blockIndex = index / GGMLType.QK_MXFP4;
        int inBlockIndex = (int) (index % GGMLType.QK_MXFP4);
        long blockOffset = blockIndex * GGMLType.MXFP4.getTypeSize();

        int e8m0 = Byte.toUnsignedInt(readByte(memorySegment, blockOffset));
        float d = e8m0ToFp32Half(e8m0);

        long qsOffset = blockOffset + Byte.BYTES + (inBlockIndex & 0x0F);
        int packed = Byte.toUnsignedInt(readByte(memorySegment, qsOffset));
        int nibble = inBlockIndex < (GGMLType.QK_MXFP4 / 2) ? (packed & 0x0F) : ((packed >>> 4) & 0x0F);
        return MXFP4_VALUES[nibble] * d;
    }

    @Override
    float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (that instanceof ArrayFloatTensor aft) {
            if (USE_VECTOR_API) {
                return vectorDot(this, thisOffset, aft, thatOffset, size);
            }
            return scalarDot(this, thisOffset, aft, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(MXFP4FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        assert Integer.bitCount(GGMLType.QK_MXFP4) == 1 : "power of 2";
        int j = 0;
        float result = 0f;

        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.QK_MXFP4 - 1));
        if (alignmentBound > 0) {
            result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j = alignmentBound;
        }

        int upperBound = j + (size - j) / GGMLType.QK_MXFP4 * GGMLType.QK_MXFP4;
        for (; j < upperBound; j += GGMLType.QK_MXFP4) {
            long blockOffset = (long) (thisOffset + j) / GGMLType.QK_MXFP4 * GGMLType.MXFP4.getTypeSize();
            float d = e8m0ToFp32Half(Byte.toUnsignedInt(readByte(thiz.memorySegment, blockOffset)));

            ByteVector packed = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + Byte.BYTES, ByteOrder.LITTLE_ENDIAN);
            ByteVector lo = packed.and((byte) 0x0F);
            ByteVector hi = packed.lanewise(VectorOperators.LSHR, 4);

            float blockSum = 0f;
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    FloatVector loCoeffs = mxfp4CodesToCoeffs((FloatVector) lo.castShape(F_SPECIES, 0));
                    FloatVector hiCoeffs = mxfp4CodesToCoeffs((FloatVector) hi.castShape(F_SPECIES, 0));
                    FloatVector xLo = that.getFloatVector(F_SPECIES, thatOffset + j);
                    FloatVector xHi = that.getFloatVector(F_SPECIES, thatOffset + j + GGMLType.QK_MXFP4 / 2);
                    blockSum += loCoeffs.fma(xLo, hiCoeffs.mul(xHi)).reduceLanes(VectorOperators.ADD);
                }
                case 256 -> {
                    FloatVector lo0 = mxfp4CodesToCoeffs((FloatVector) lo.castShape(F_SPECIES, 0));
                    FloatVector lo1 = mxfp4CodesToCoeffs((FloatVector) lo.castShape(F_SPECIES, 1));
                    FloatVector hi0 = mxfp4CodesToCoeffs((FloatVector) hi.castShape(F_SPECIES, 0));
                    FloatVector hi1 = mxfp4CodesToCoeffs((FloatVector) hi.castShape(F_SPECIES, 1));
                    FloatVector x0 = that.getFloatVector(F_SPECIES, thatOffset + j);
                    FloatVector x1 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length());
                    FloatVector x2 = that.getFloatVector(F_SPECIES, thatOffset + j + GGMLType.QK_MXFP4 / 2);
                    FloatVector x3 = that.getFloatVector(F_SPECIES, thatOffset + j + GGMLType.QK_MXFP4 / 2 + F_SPECIES.length());
                    blockSum += lo0.fma(x0, lo1.mul(x1)).reduceLanes(VectorOperators.ADD);
                    blockSum += hi0.fma(x2, hi1.mul(x3)).reduceLanes(VectorOperators.ADD);
                }
                case 128 -> {
                    FloatVector sum = FloatVector.zero(F_SPECIES);
                    for (int p = 0; p < 4; p++) {
                        FloatVector loPart = mxfp4CodesToCoeffs((FloatVector) lo.castShape(F_SPECIES, p));
                        FloatVector hiPart = mxfp4CodesToCoeffs((FloatVector) hi.castShape(F_SPECIES, p));
                        FloatVector xLo = that.getFloatVector(F_SPECIES, thatOffset + j + p * F_SPECIES.length());
                        FloatVector xHi = that.getFloatVector(F_SPECIES, thatOffset + j + GGMLType.QK_MXFP4 / 2 + p * F_SPECIES.length());
                        sum = loPart.fma(xLo, sum);
                        sum = hiPart.fma(xHi, sum);
                    }
                    blockSum += sum.reduceLanes(VectorOperators.ADD);
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }

            result += blockSum * d;
        }

        if (j < size) {
            result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }
        return result;
    }

    private static FloatVector mxfp4CodesToCoeffs(FloatVector codes) {
        FloatVector zero = FloatVector.zero(F_SPECIES);
        FloatVector eight = FloatVector.broadcast(F_SPECIES, 8f);
        var negMask = codes.compare(VectorOperators.GE, 8f);

        FloatVector t = codes.sub(zero.blend(eight, negMask));
        FloatVector mag = t
                .add(t.sub(4f).lanewise(VectorOperators.MAX, 0f))
                .add(t.sub(6f).lanewise(VectorOperators.MAX, 0f).mul(2f));
        return mag.blend(mag.neg(), negMask);
    }

    private static float scalarDot(MXFP4FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        for (int i = 0; i < size; i++) {
            result += thiz.getFloat(thisOffset + i) * that.values[thatOffset + i];
        }
        return result;
    }

    private static float e8m0ToFp32Half(int x) {
        int bits;
        if (x < 2) {
            bits = 0x00200000 << x;
        } else {
            bits = (x - 1) << 23;
        }
        return Float.intBitsToFloat(bits);
    }
}

final class ModelLoader {
    public static FloatTensor loadQuantized(GGMLTensorEntry entry) {
        GGMLType ggmlType = entry.ggmlType();
        long numElements = FloatTensor.numberOfElementsLong(entry.shape());
        return switch (ggmlType) {
            case Q8_0 -> new Q8_0FloatTensor(numElements, entry.memorySegment());
            case Q4_0 -> new Q4_0FloatTensor(numElements, entry.memorySegment());
            case Q4_1 -> new Q4_1FloatTensor(numElements, entry.memorySegment());
            case Q4_K -> new Q4_KFloatTensor(numElements, entry.memorySegment());
            case Q5_K -> new Q5_KFloatTensor(numElements, entry.memorySegment());
            case Q6_K -> new Q6_KFloatTensor(numElements, entry.memorySegment());
            case F32 -> new F32FloatTensor(numElements, entry.memorySegment());
            case F16 -> new F16FloatTensor(numElements, entry.memorySegment());
            case BF16 -> new BF16FloatTensor(numElements, entry.memorySegment());
            case MXFP4 -> new MXFP4FloatTensor(numElements, entry.memorySegment());
            default -> throw new UnsupportedOperationException("Quantization format " + ggmlType);
        };
    }

    public static FloatTensor[] loadArrayOfQuantized(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatTensor[] array = new FloatTensor[size];
        for (int i = 0; i < size; i++) array[i] = loadQuantized(getTensorEntry.apply(i));
        return array;
    }

    public static FloatBuffer[] loadArrayOfFloatBuffer(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatBuffer[] array = new FloatBuffer[size];
        for (int i = 0; i < size; i++) array[i] = toFloatBuffer(getTensorEntry.apply(i));
        return array;
    }

    public static FloatBuffer toFloatBuffer(GGMLTensorEntry tensorEntry) {
        if (tensorEntry.ggmlType() != GGMLType.F32) throw new UnsupportedOperationException("Conversion to " + tensorEntry.ggmlType());
        return tensorEntry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
    }

    public static Qwen35 loadModel(Path ggufPath, int contextLength) throws IOException {
        try (var ignored = Timer.log("Load Qwen35 model")) {
            try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
                GGUF gguf = GGUF.loadModel(fileChannel, ggufPath.toString());
                return loadModel(fileChannel, gguf, contextLength);
            }
        }
    }

    static Qwen35 loadModel(FileChannel fileChannel, GGUF gguf, int contextLength) throws IOException {
        Map<String, Object> metadata = gguf.getMetadata();
        // Detect architecture - support both qwen35 and qwen35moe
        String arch = (String) metadata.getOrDefault("general.architecture", "qwen35");
        if (!arch.equals("qwen35") && !arch.equals("qwen35moe")) {
            throw new IllegalArgumentException("Unsupported architecture: " + arch);
        }
        boolean isMoE = arch.equals("qwen35moe");
        int embeddingLength = (int) metadata.get(arch + ".embedding_length");
        int numberOfLayers = (int) metadata.get(arch + ".block_count");
        int numberOfHeads = (int) metadata.get(arch + ".attention.head_count");
        int numberOfKeyValueHeads = (int) metadata.getOrDefault(arch + ".attention.head_count_kv", numberOfHeads);
        int contextLengthCfg = (int) metadata.get(arch + ".context_length");
        if (contextLength < 0 || contextLengthCfg < contextLength) contextLength = contextLengthCfg;
        float rmsNormEps = (float) metadata.getOrDefault(arch + ".attention.layer_norm_rms_epsilon", 1e-6f);
        float ropeTheta = (float) metadata.getOrDefault(arch + ".rope.freq_base", 1000000f);
        int headSize = (int) metadata.getOrDefault(arch + ".attention.key_length", embeddingLength / numberOfHeads);
        int hiddenDim;
        if (isMoE) {
            // For MoE models, use expert_feed_forward_length
            hiddenDim = ((Number) metadata.getOrDefault(arch + ".expert_feed_forward_length", 5632)).intValue();
        } else {
            // For dense models, use feed_forward_length
            hiddenDim = (int) metadata.get(arch + ".feed_forward_length");
        }
        
        // MoE configuration fields
        int expertCount = isMoE ? ((Number) metadata.getOrDefault(arch + ".expert_count", 1)).intValue() : 1;
        int expertUsedCount = isMoE ? ((Number) metadata.getOrDefault(arch + ".expert_used_count", 1)).intValue() : 1;
        int expertFeedForwardLength = isMoE ? ((Number) metadata.getOrDefault(arch + ".expert_feed_forward_length", hiddenDim)).intValue() : hiddenDim;
        int expertSharedFeedForwardLength = isMoE ? ((Number) metadata.getOrDefault(arch + ".expert_shared_feed_forward_length", 0)).intValue() : 0;
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        int vocabularySize = tokens.length;
        int fullAttentionInterval = (int) metadata.getOrDefault(arch + ".full_attention_interval", 4);
        int[] ropeDimensionSections = (int[]) metadata.getOrDefault(arch + ".rope.dimension_sections", new int[]{5, 6, 7, 8});
        int ropeDimensionCount = (int) metadata.getOrDefault(arch + ".rope.dimension_count",
                Arrays.stream(ropeDimensionSections).sum() * 2);
        int ssmConvKernel = (int) metadata.getOrDefault(arch + ".ssm.conv_kernel", 4);
        int ssmGroupCount = (int) metadata.getOrDefault(arch + ".ssm.group_count", 8);
        int ssmInnerSize = (int) metadata.getOrDefault(arch + ".ssm.inner_size", 2048);
        int ssmStateSize = (int) metadata.getOrDefault(arch + ".ssm.state_size", 128);
        int ssmTimeStepRank = (int) metadata.getOrDefault(arch + ".ssm.time_step_rank", 16);
        boolean[] isFullAttention = new boolean[numberOfLayers];
        for (int i = 0; i < numberOfLayers; i++) isFullAttention[i] = (i + 1) % fullAttentionInterval == 0;

        Qwen35.Configuration config = new Qwen35.Configuration(embeddingLength, numberOfLayers, numberOfHeads, numberOfKeyValueHeads,
                contextLength, rmsNormEps, ropeTheta, headSize, hiddenDim, vocabularySize, fullAttentionInterval, isFullAttention,
                ssmConvKernel, ssmGroupCount, ssmInnerSize, ssmStateSize, ssmTimeStepRank, ropeDimensionSections, ropeDimensionCount,
                isMoE, expertCount, expertUsedCount, expertFeedForwardLength, expertSharedFeedForwardLength);

        Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
        Qwen35.Weights weights = loadWeights(tensorEntries, config);
        float[] scores = (float[]) metadata.getOrDefault("tokenizer.ggml.scores", new float[tokens.length]);
        Vocabulary vocabulary = new Vocabulary(tokens, scores);
        String[] merges = (String[]) metadata.getOrDefault("tokenizer.ggml.merges", new String[0]);
        int[] tokenType = (int[]) metadata.getOrDefault("tokenizer.ggml.token_type", new int[tokens.length]);
        return new Qwen35(config, weights, vocabulary, merges, tokenType);
    }

    public static Qwen35.Weights loadWeights(Map<String, GGMLTensorEntry> tensorEntries, Qwen35.Configuration config) {
        int nLayers = config.numberOfLayers;
        FloatTensor tokenEmbeddingTable = loadQuantized(tensorEntries.get("token_embd.weight"));
        FloatBuffer outputNorm = toFloatBuffer(tensorEntries.get("output_norm.weight"));
        FloatTensor outputWeight = tensorEntries.containsKey("output.weight") ? loadQuantized(tensorEntries.get("output.weight")) : tokenEmbeddingTable;
        FloatBuffer[] attnNorm = new FloatBuffer[nLayers];
        FloatTensor[] attnQ = new FloatTensor[nLayers];
        FloatTensor[] attnK = new FloatTensor[nLayers];
        FloatTensor[] attnV = new FloatTensor[nLayers];
        FloatTensor[] attnOutput = new FloatTensor[nLayers];
        FloatBuffer[] attnQNorm = new FloatBuffer[nLayers];
        FloatBuffer[] attnKNorm = new FloatBuffer[nLayers];
        FloatTensor[] attnQkv = new FloatTensor[nLayers];
        FloatTensor[] attnGate = new FloatTensor[nLayers];
        FloatBuffer[] ssmA = new FloatBuffer[nLayers];
        FloatTensor[] ssmAlpha = new FloatTensor[nLayers];
        FloatTensor[] ssmBeta = new FloatTensor[nLayers];
        FloatBuffer[] ssmConv1d = new FloatBuffer[nLayers];
        FloatBuffer[] ssmDtBias = new FloatBuffer[nLayers];
        FloatBuffer[] ssmNorm = new FloatBuffer[nLayers];
        FloatTensor[] ssmOut = new FloatTensor[nLayers];
        FloatBuffer[] ffnNorm = new FloatBuffer[nLayers];
        FloatTensor[] ffnGate = new FloatTensor[nLayers];
        FloatTensor[] ffnDown = new FloatTensor[nLayers];
        FloatTensor[] ffnUp = new FloatTensor[nLayers];
        FloatBuffer[] postAttentionNorm = new FloatBuffer[nLayers];
        
        // MoE weights (null for dense models)
        FloatTensor[] moeExpertGate = config.isMoE ? new FloatTensor[nLayers] : null;
        FloatTensor[] moeExpertUp = config.isMoE ? new FloatTensor[nLayers] : null;
        FloatTensor[] moeExpertDown = config.isMoE ? new FloatTensor[nLayers] : null;
        FloatTensor[] moeSharedGate = config.isMoE ? new FloatTensor[nLayers] : null;
        FloatTensor[] moeSharedUp = config.isMoE ? new FloatTensor[nLayers] : null;
        FloatTensor[] moeSharedDown = config.isMoE ? new FloatTensor[nLayers] : null;
        FloatTensor[] moeSharedInputGate = config.isMoE ? new FloatTensor[nLayers] : null;
        FloatTensor[] moeRouter = config.isMoE ? new FloatTensor[nLayers] : null;

        for (int i = 0; i < nLayers; i++) {
            String prefix = "blk." + i + ".";
            attnNorm[i] = toFloatBuffer(tensorEntries.get(prefix + "attn_norm.weight"));
            ffnNorm[i] = toFloatBuffer(tensorEntries.get(prefix + "post_attention_norm.weight"));
            postAttentionNorm[i] = toFloatBuffer(tensorEntries.get(prefix + "post_attention_norm.weight"));
            
            if (config.isMoE) {
                // Load MoE expert weights: prefer consolidated tensors used by qwen35moe,
                // but keep fallback support for per-expert tensors.
                GGMLTensorEntry gateExps = tensorEntries.get(prefix + "ffn_gate_exps.weight");
                GGMLTensorEntry upExps = tensorEntries.get(prefix + "ffn_up_exps.weight");
                GGMLTensorEntry downExps = tensorEntries.get(prefix + "ffn_down_exps.weight");
                if (gateExps != null && upExps != null && downExps != null) {
                    moeExpertGate[i] = loadQuantized(gateExps);
                    moeExpertUp[i] = loadQuantized(upExps);
                    moeExpertDown[i] = loadQuantized(downExps);
                } else {
                    FloatTensor gatePacked = ArrayFloatTensor.allocate(config.expertCount * config.expertFeedForwardLength, config.embeddingLength);
                    FloatTensor upPacked = ArrayFloatTensor.allocate(config.expertCount * config.expertFeedForwardLength, config.embeddingLength);
                    FloatTensor downPacked = ArrayFloatTensor.allocate(config.expertCount * config.embeddingLength, config.expertFeedForwardLength);
                    for (int e = 0; e < config.expertCount; e++) {
                        String expertPrefix = prefix + "ffn_experts." + e + ".";
                        FloatTensor gate = loadQuantized(tensorEntries.get(expertPrefix + "gate.weight"));
                        FloatTensor up = loadQuantized(tensorEntries.get(expertPrefix + "up.weight"));
                        FloatTensor down = loadQuantized(tensorEntries.get(expertPrefix + "down.weight"));
                        gate.copyTo(0, gatePacked, e * config.expertFeedForwardLength * config.embeddingLength,
                                config.expertFeedForwardLength * config.embeddingLength);
                        up.copyTo(0, upPacked, e * config.expertFeedForwardLength * config.embeddingLength,
                                config.expertFeedForwardLength * config.embeddingLength);
                        down.copyTo(0, downPacked, e * config.embeddingLength * config.expertFeedForwardLength,
                                config.embeddingLength * config.expertFeedForwardLength);
                    }
                    moeExpertGate[i] = gatePacked;
                    moeExpertUp[i] = upPacked;
                    moeExpertDown[i] = downPacked;
                }
                // Load shared expert weights.
                if (config.expertSharedFeedForwardLength > 0) {
                    GGMLTensorEntry sharedGate = tensorEntries.get(prefix + "ffn_gate_shexp.weight");
                    GGMLTensorEntry sharedUp = tensorEntries.get(prefix + "ffn_up_shexp.weight");
                    GGMLTensorEntry sharedDown = tensorEntries.get(prefix + "ffn_down_shexp.weight");
                    if (sharedGate == null || sharedUp == null || sharedDown == null) {
                        sharedGate = tensorEntries.get(prefix + "ffn_shared_expert_gate.weight");
                        sharedUp = tensorEntries.get(prefix + "ffn_shared_expert_up.weight");
                        sharedDown = tensorEntries.get(prefix + "ffn_shared_expert_down.weight");
                    }
                    if (sharedGate != null && sharedUp != null && sharedDown != null) {
                        moeSharedGate[i] = loadQuantized(sharedGate);
                        moeSharedUp[i] = loadQuantized(sharedUp);
                        moeSharedDown[i] = loadQuantized(sharedDown);
                    }
                    GGMLTensorEntry sharedInpGate = tensorEntries.get(prefix + "ffn_gate_inp_shexp.weight");
                    if (sharedInpGate == null) {
                        sharedInpGate = tensorEntries.get(prefix + "ffn_shared_expert_gate_inp.weight");
                    }
                    if (sharedInpGate != null) {
                        moeSharedInputGate[i] = loadQuantized(sharedInpGate);
                    }
                }
                // Load router.
                GGMLTensorEntry router = tensorEntries.get(prefix + "ffn_gate_inp.weight");
                if (router == null) {
                    router = tensorEntries.get(prefix + "ffn_router.weight");
                }
                moeRouter[i] = loadQuantized(router);
            } else {
                // Load dense FFN weights
                ffnGate[i] = loadQuantized(tensorEntries.get(prefix + "ffn_gate.weight"));
                ffnDown[i] = loadQuantized(tensorEntries.get(prefix + "ffn_down.weight"));
                ffnUp[i] = loadQuantized(tensorEntries.get(prefix + "ffn_up.weight"));
            }
            
            if (config.isFullAttention[i]) {
                attnQ[i] = loadQuantized(tensorEntries.get(prefix + "attn_q.weight"));
                attnK[i] = loadQuantized(tensorEntries.get(prefix + "attn_k.weight"));
                attnV[i] = loadQuantized(tensorEntries.get(prefix + "attn_v.weight"));
                attnOutput[i] = loadQuantized(tensorEntries.get(prefix + "attn_output.weight"));
                attnQNorm[i] = toFloatBuffer(tensorEntries.get(prefix + "attn_q_norm.weight"));
                attnKNorm[i] = toFloatBuffer(tensorEntries.get(prefix + "attn_k_norm.weight"));
            } else {
                attnQkv[i] = loadQuantized(tensorEntries.get(prefix + "attn_qkv.weight"));
                attnGate[i] = loadQuantized(tensorEntries.get(prefix + "attn_gate.weight"));
                ssmA[i] = toFloatBuffer(tensorEntries.get(prefix + "ssm_a"));
                ssmAlpha[i] = loadQuantized(tensorEntries.get(prefix + "ssm_alpha.weight"));
                ssmBeta[i] = loadQuantized(tensorEntries.get(prefix + "ssm_beta.weight"));
                ssmConv1d[i] = toFloatBuffer(tensorEntries.get(prefix + "ssm_conv1d.weight"));
                ssmDtBias[i] = toFloatBuffer(tensorEntries.get(prefix + "ssm_dt.bias"));
                ssmNorm[i] = toFloatBuffer(tensorEntries.get(prefix + "ssm_norm.weight"));
                ssmOut[i] = loadQuantized(tensorEntries.get(prefix + "ssm_out.weight"));
            }
        }
        return new Qwen35.Weights(tokenEmbeddingTable, outputNorm, outputWeight, attnNorm, attnQ, attnK, attnV, attnOutput, attnQNorm, attnKNorm,
                attnQkv, attnGate, ssmA, ssmAlpha, ssmBeta, ssmConv1d, ssmDtBias, ssmNorm, ssmOut, ffnNorm, ffnGate, ffnDown, ffnUp, postAttentionNorm,
                moeExpertGate, moeExpertUp, moeExpertDown, moeSharedGate, moeSharedUp, moeSharedDown, moeSharedInputGate, moeRouter, config.isMoE);
    }
}

record Vocabulary(String[] tokens, float[] scores, Map<String, Integer> tokenToIndex) {
    public Vocabulary(String[] vocabulary, float[] scores) {
        this(vocabulary, scores, IntStream.range(0, vocabulary.length).boxed().collect(Collectors.toMap(i -> vocabulary[i], i -> i)));
    }
    public String get(int tokenIndex) { return tokens[tokenIndex]; }
    public OptionalInt getIndex(String token) {
        Integer value = tokenToIndex.get(token);
        return value != null ? OptionalInt.of(value) : OptionalInt.empty();
    }
    public int size() { return tokens.length; }
    public float getScore(int tokenIndex) { return scores[tokenIndex]; }
}

/**
 * GPT-2 style BPE tokenizer, adapted from Llama3.java for Qwen models.
 * <a href="https://github.com/openai/gpt-2/blob/master/src/encoder.py">GPT 2 tokenizer</a>
 */
final class QwenTokenizer {
    private static final String QWEN_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)" +
            "|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+" +
            "|\\p{N}" +
            "| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*" +
            "|\\s*[\\r\\n]+" +
            "|\\s+(?!\\S)" +
            "|\\s+";

    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    private final Map<Pair<Integer, Integer>, Integer> merges;
    private final Map<String, Integer> specialTokens;

    public QwenTokenizer(Vocabulary vocabulary, String[] mergeLines, int[] tokenType) {
        this.vocabulary = vocabulary;
        this.compiledPattern = Pattern.compile(QWEN_PATTERN);
        // Build merge ranks as token index pairs (like Llama3)
        this.merges = HashMap.newHashMap(mergeLines.length);
        for (Pair<Integer, Integer> pair : Arrays.stream(mergeLines)
                .map(line -> line.split(" "))
                .map(parts -> new Pair<>(
                        vocabulary.getIndex(parts[0]).orElseThrow(),
                        vocabulary.getIndex(parts[1]).orElseThrow()))
                .toList()) {
            int mergeIndex = vocabulary.getIndex(vocabulary.get(pair.first()) + vocabulary.get(pair.second())).orElseThrow();
            this.merges.put(pair, mergeIndex);
        }
        // Build special tokens map (type != 1 means special/control)
        this.specialTokens = new HashMap<>();
        for (int i = 0; i < tokenType.length; i++) {
            if (tokenType[i] != 1) specialTokens.put(vocabulary.get(i), i);
        }
    }

    public Vocabulary vocabulary() { return vocabulary; }
    public Map<String, Integer> specialTokens() { return specialTokens; }
    public int getEosToken() { return specialTokens.getOrDefault("<|im_end|>", specialTokens.getOrDefault("<|endoftext|>", 0)); }
    public boolean isSpecialToken(int tokenIndex) { return specialTokens.containsValue(tokenIndex); }

    /**
     * Returns list of utf-8 byte and a corresponding list of unicode strings.
     * Avoids mapping to whitespace/control characters the bpe code barfs on.
     */
    private static Map<Integer, Integer> bytesToUnicode() {
        List<Integer> bs = new ArrayList<>();
        IntStream.rangeClosed('!', '~').forEach(bs::add);
        IntStream.rangeClosed('\u00A1', '\u00AC').forEach(bs::add);
        IntStream.rangeClosed('\u00AE', '\u00FF').forEach(bs::add);
        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n += 1;
            }
        }
        return IntStream.range(0, bs.size())
                .boxed()
                .collect(Collectors.toMap(bs::get, cs::get));
    }

    static final Map<Integer, Integer> BYTE_ENCODER = bytesToUnicode();
    static final Map<Integer, Integer> BYTE_DECODER = BYTE_ENCODER.entrySet()
            .stream()
            .collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));

    /** Encode text to token IDs (byte-encoding + BPE), no special token handling. */
    public int[] encode(String text) {
        StringBuilder sb = new StringBuilder();
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        for (byte b : bytes) {
            sb.appendCodePoint(BYTE_ENCODER.get(Byte.toUnsignedInt(b)));
        }
        return encodeImpl(sb.toString());
    }

    public List<Integer> encodeAsList(String text) {
        return Arrays.stream(encode(text)).boxed().toList();
    }

    private int[] encodeImpl(String text) {
        return encodeOrdinary(text).stream().mapToInt(i -> i).toArray();
    }

    private List<Integer> encodeOrdinary(String text) {
        List<String> textChunks = findAll(compiledPattern, text);
        List<Integer> ids = new ArrayList<>();
        for (String chunk : textChunks) {
            ids.addAll(encodeChunk(chunk));
        }
        return ids;
    }

    private static List<String> findAll(Pattern pattern, String text) {
        List<String> allMatches = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        while (matcher.find()) {
            allMatches.add(matcher.group());
        }
        return allMatches;
    }

    private Map<Pair<Integer, Integer>, Integer> getStats(List<Integer> ids) {
        Map<Pair<Integer, Integer>, Integer> map = new HashMap<>();
        for (int i = 0; i + 1 < ids.size(); i++) {
            Pair<Integer, Integer> key = new Pair<>(ids.get(i), ids.get(i + 1));
            map.put(key, map.getOrDefault(key, 0) + 1);
        }
        return map;
    }

    private List<Integer> encodeChunk(String chunk) {
        List<Integer> ids = new ArrayList<>();
        for (int b : chunk.toCharArray()) {
            int tokenIndex = this.vocabulary.getIndex(String.valueOf((char) b)).orElseThrow();
            ids.add(tokenIndex);
        }
        while (ids.size() >= 2) {
            Map<Pair<Integer, Integer>, Integer> stats = getStats(ids);
            Pair<Integer, Integer> pair = stats.keySet().stream()
                    .min(Comparator.comparingInt(key -> this.merges.getOrDefault(key, Integer.MAX_VALUE)))
                    .orElseThrow();
            if (!this.merges.containsKey(pair)) {
                break;
            }
            int idx = this.merges.get(pair);
            ids = merge(ids, pair, idx);
        }
        return ids;
    }

    private static List<Integer> merge(List<Integer> ids, Pair<Integer, Integer> pair, int idx) {
        List<Integer> newids = new ArrayList<>();
        int i = 0;
        while (i < ids.size()) {
            if (ids.get(i).equals(pair.first()) && i < ids.size() - 1 && ids.get(i + 1).equals(pair.second())) {
                newids.add(idx);
                i += 2;
            } else {
                newids.add(ids.get(i));
                i += 1;
            }
        }
        return newids;
    }

    /** Decode token IDs back to text, properly handling multi-byte UTF-8. */
    public String decode(List<Integer> tokens) {
        StringBuilder result = new StringBuilder();
        List<Integer> pendingBPE = new ArrayList<>();
        for (int token : tokens) {
            if (isSpecialToken(token)) {
                // Flush pending BPE tokens through byte decoder
                if (!pendingBPE.isEmpty()) {
                    result.append(decodeBPE(pendingBPE));
                    pendingBPE.clear();
                }
                result.append(vocabulary.get(token));
            } else {
                pendingBPE.add(token);
            }
        }
        if (!pendingBPE.isEmpty()) {
            result.append(decodeBPE(pendingBPE));
        }
        return result.toString();
    }

    private String decodeBPE(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            sb.append(vocabulary.get(token));
        }
        String decoded = sb.toString();
        int[] decodedBytesAsInts = decoded.codePoints().map(BYTE_DECODER::get).toArray();
        byte[] rawBytes = new byte[decodedBytesAsInts.length];
        for (int i = 0; i < rawBytes.length; i++) {
            rawBytes[i] = (byte) decodedBytesAsInts[i];
        }
        return new String(rawBytes, StandardCharsets.UTF_8);
    }

    public static String replaceControlCharacters(String str) {
        StringBuilder chars = new StringBuilder();
        for (int cp : str.codePoints().toArray()) {
            if (Character.getType(cp) == Character.CONTROL && cp != '\n') {
                chars.append("\\u").append(HexFormat.of().toHexDigits(cp, 4));
            } else {
                chars.appendCodePoint(cp);
            }
        }
        return chars.toString();
    }
}

class QwenChatFormat {
    protected final QwenTokenizer tokenizer;
    private final int imStart;
    private final int imEnd;
    private final int thinkOpen;
    private final int thinkClose;

    public QwenChatFormat(QwenTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specials = tokenizer.specialTokens();
        this.imStart = specials.get("<|im_start|>");
        this.imEnd = specials.get("<|im_end|>");
        this.thinkOpen = specials.getOrDefault("<think>", -1);
        this.thinkClose = specials.getOrDefault("</think>", -1);
    }

    public Set<Integer> getStopTokens() {
        Set<Integer> tokens = new HashSet<>();
        tokens.add(tokenizer.getEosToken());
        tokens.add(imEnd);
        return tokens;
    }

    public List<Integer> encodeMessage(String role, String content) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(imStart);
        tokens.addAll(tokenizer.encodeAsList(role + "\n" + content));
        tokens.add(imEnd);
        tokens.addAll(tokenizer.encodeAsList("\n"));
        return tokens;
    }

    public List<Integer> encodeHeader(String role, boolean think) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(imStart);
        tokens.addAll(tokenizer.encodeAsList(role + "\n"));
        if ("assistant".equals(role) && !think && thinkOpen >= 0 && thinkClose >= 0) {
            // Add empty think block to disable thinking mode
            tokens.add(thinkOpen);
            tokens.addAll(tokenizer.encodeAsList("\n\n"));
            tokens.add(thinkClose);
            tokens.addAll(tokenizer.encodeAsList("\n\n"));
        }
        return tokens;
    }
}

@FunctionalInterface
interface Sampler {
    int sampleToken(FloatTensor logits);
    Sampler ARGMAX = FloatTensor::argmax;
}

record CategoricalSampler(RandomGenerator rng) implements Sampler {
    @Override
    public int sampleToken(FloatTensor logits) {
        float random0to1 = rng.nextFloat(1f);
        float cdf = 0.0f;
        for (int i = 0; i < logits.size(); i++) {
            cdf += logits.getFloat(i);
            if (random0to1 < cdf) return i;
        }
        return Math.toIntExact(logits.size()) - 1;
    }
}

final class ToppSampler implements Sampler {
    final int[] indices;
    final float topp;
    final RandomGenerator rng;
    public ToppSampler(int maxNumberOfElements, float topp, RandomGenerator rng) {
        this.indices = new int[maxNumberOfElements];
        this.topp = topp;
        this.rng = rng;
    }
    static void swap(int[] array, int from, int to) {
        int tmp = array[from];
        array[from] = array[to];
        array[to] = tmp;
    }
    static void siftDown(int[] array, int from, int n, Comparator<Integer> comparator) {
        int prev = from, next;
        while ((next = 2 * prev + 1) < n) {
            int r = 2 * prev + 2;
            if (r < n && comparator.compare(array[r], array[next]) < 0) next = r;
            if (comparator.compare(array[next], array[prev]) < 0) {
                swap(array, prev, next);
                prev = next;
            } else break;
        }
    }
    @Override
    public int sampleToken(FloatTensor logits) {
        Comparator<Integer> comparator = Comparator.comparingDouble((Integer i) -> logits.getFloat(i)).reversed();
        int n = Math.toIntExact(logits.size());
        int head = 0, tail = n - 1;
        float cutoff = (1.0f - topp) / (n - 1);
        for (int i = 0; i < indices.length; i++) {
            if (logits.getFloat(i) >= cutoff) indices[head++] = i;
            else indices[tail--] = i;
        }
        int n0 = head;
        for (int i = n0 / 2 - 1; i >= 0; --i) siftDown(indices, i, n0, comparator);
        float cumulativeProb = 0.0f;
        int lastIndex = 0;
        for (int i = n0 - 1; i >= 0; i--) {
            swap(indices, 0, i);
            cumulativeProb += logits.getFloat(indices[i]);
            if (cumulativeProb > topp) {
                lastIndex = i;
                break;
            }
            siftDown(indices, 0, i, comparator);
        }
        float r = rng.nextFloat(1f) * cumulativeProb;
        float cdf = 0.0f;
        for (int i = n0 - 1; i >= lastIndex; i--) {
            cdf += logits.getFloat(indices[i]);
            if (r < cdf) return indices[i];
        }
        return indices[lastIndex];
    }
}

public class Qwen35 {
    public final Configuration configuration;
    public final Weights weights;
    public final QwenTokenizer tokenizer;
    
    public Qwen35(Configuration configuration, Weights weights, Vocabulary vocabulary, String[] merges, int[] tokenType) {
        this.configuration = configuration;
        this.weights = weights;
        this.tokenizer = new QwenTokenizer(vocabulary, merges, tokenType);
    }
    
    public State createNewState() {
        State state = new State(configuration);
        state.latestToken = -1;
        return state;
    }
    
    public static final class Configuration {
        public final int embeddingLength, numberOfLayers, numberOfHeads, numberOfKeyValueHeads;
        public final int contextLength;
        public final float rmsNormEps, ropeTheta;
        public final int headSize, hiddenDim, vocabularySize;
        public final int fullAttentionInterval;
        public final boolean[] isFullAttention;
        public final int ssmConvKernel, ssmGroupCount, ssmInnerSize, ssmStateSize, ssmTimeStepRank;
        public final int[] ropeSections;
        public final int ropeDimensionCount;
        // MoE configuration
        public final boolean isMoE;
        public final int expertCount, expertUsedCount, expertFeedForwardLength, expertSharedFeedForwardLength;
        
        public Configuration(int embeddingLength, int numberOfLayers, int numberOfHeads, int numberOfKeyValueHeads,
                             int contextLength, float rmsNormEps, float ropeTheta, int headSize, int hiddenDim,
                             int vocabularySize, int fullAttentionInterval, boolean[] isFullAttention,
                             int ssmConvKernel, int ssmGroupCount, int ssmInnerSize, int ssmStateSize,
                             int ssmTimeStepRank, int[] ropeSections, int ropeDimensionCount,
                             boolean isMoE, int expertCount, int expertUsedCount, 
                             int expertFeedForwardLength, int expertSharedFeedForwardLength) {
            this.embeddingLength = embeddingLength;
            this.numberOfLayers = numberOfLayers;
            this.numberOfHeads = numberOfHeads;
            this.numberOfKeyValueHeads = numberOfKeyValueHeads;
            this.contextLength = contextLength;
            this.rmsNormEps = rmsNormEps;
            this.ropeTheta = ropeTheta;
            this.headSize = headSize;
            this.hiddenDim = hiddenDim;
            this.vocabularySize = vocabularySize;
            this.fullAttentionInterval = fullAttentionInterval;
            this.isFullAttention = isFullAttention;
            this.ssmConvKernel = ssmConvKernel;
            this.ssmGroupCount = ssmGroupCount;
            this.ssmInnerSize = ssmInnerSize;
            this.ssmStateSize = ssmStateSize;
            this.ssmTimeStepRank = ssmTimeStepRank;
            this.ropeSections = ropeSections;
            this.ropeDimensionCount = ropeDimensionCount;
            this.isMoE = isMoE;
            this.expertCount = expertCount;
            this.expertUsedCount = expertUsedCount;
            this.expertFeedForwardLength = expertFeedForwardLength;
            this.expertSharedFeedForwardLength = expertSharedFeedForwardLength;
        }
        
        public int queryDim() { return numberOfHeads * headSize; }
        public int kvDim() { return numberOfKeyValueHeads * headSize; }
        public int convChannels() { return ssmInnerSize + 2 * ssmGroupCount * ssmStateSize; }
        public boolean isMoE() { return isMoE; }
    }
    
    public static final class Weights {
        public final FloatTensor tokenEmbeddingTable;
        public final FloatBuffer outputNorm;
        public final FloatTensor outputWeight;
        public final FloatBuffer[] attnNorm;
        public final FloatTensor[] attnQ, attnK, attnV, attnOutput;
        public final FloatBuffer[] attnQNorm, attnKNorm;
        public final FloatTensor[] attnQkv, attnGate;
        public final FloatBuffer[] ssmA;
        public final FloatTensor[] ssmAlpha, ssmBeta;
        public final FloatBuffer[] ssmConv1d, ssmDtBias, ssmNorm;
        public final FloatTensor[] ssmOut;
        public final FloatBuffer[] ffnNorm, postAttentionNorm;
        public final FloatTensor[] ffnGate, ffnDown, ffnUp;
        // MoE weights
        public final FloatTensor[] moeExpertGate;      // [layer], packed as [expert][expertFFN][dim]
        public final FloatTensor[] moeExpertUp;        // [layer], packed as [expert][expertFFN][dim]
        public final FloatTensor[] moeExpertDown;      // [layer], packed as [expert][dim][expertFFN]
        public final FloatTensor[] moeSharedGate;      // [layer]
        public final FloatTensor[] moeSharedUp;        // [layer]
        public final FloatTensor[] moeSharedDown;      // [layer]
        public final FloatTensor[] moeSharedInputGate; // [layer], scalar gate per token
        public final FloatTensor[] moeRouter;          // [layer]
        public final boolean isMoE;
        
        public Weights(FloatTensor tokenEmbeddingTable, FloatBuffer outputNorm, FloatTensor outputWeight,
                       FloatBuffer[] attnNorm, FloatTensor[] attnQ, FloatTensor[] attnK, FloatTensor[] attnV,
                       FloatTensor[] attnOutput, FloatBuffer[] attnQNorm, FloatBuffer[] attnKNorm,
                       FloatTensor[] attnQkv, FloatTensor[] attnGate, FloatBuffer[] ssmA, FloatTensor[] ssmAlpha,
                       FloatTensor[] ssmBeta, FloatBuffer[] ssmConv1d, FloatBuffer[] ssmDtBias, FloatBuffer[] ssmNorm,
                       FloatTensor[] ssmOut, FloatBuffer[] ffnNorm, FloatTensor[] ffnGate, FloatTensor[] ffnDown,
                       FloatTensor[] ffnUp, FloatBuffer[] postAttentionNorm,
                       FloatTensor[] moeExpertGate, FloatTensor[] moeExpertUp, FloatTensor[] moeExpertDown,
                       FloatTensor[] moeSharedGate, FloatTensor[] moeSharedUp, FloatTensor[] moeSharedDown,
                       FloatTensor[] moeSharedInputGate, FloatTensor[] moeRouter, boolean isMoE) {
            this.tokenEmbeddingTable = tokenEmbeddingTable;
            this.outputNorm = outputNorm;
            this.outputWeight = outputWeight;
            this.attnNorm = attnNorm;
            this.attnQ = attnQ;
            this.attnK = attnK;
            this.attnV = attnV;
            this.attnOutput = attnOutput;
            this.attnQNorm = attnQNorm;
            this.attnKNorm = attnKNorm;
            this.attnQkv = attnQkv;
            this.attnGate = attnGate;
            this.ssmA = ssmA;
            this.ssmAlpha = ssmAlpha;
            this.ssmBeta = ssmBeta;
            this.ssmConv1d = ssmConv1d;
            this.ssmDtBias = ssmDtBias;
            this.ssmNorm = ssmNorm;
            this.ssmOut = ssmOut;
            this.ffnNorm = ffnNorm;
            this.postAttentionNorm = postAttentionNorm;
            this.ffnGate = ffnGate;
            this.ffnDown = ffnDown;
            this.ffnUp = ffnUp;
            this.moeExpertGate = moeExpertGate;
            this.moeExpertUp = moeExpertUp;
            this.moeExpertDown = moeExpertDown;
            this.moeSharedGate = moeSharedGate;
            this.moeSharedUp = moeSharedUp;
            this.moeSharedDown = moeSharedDown;
            this.moeSharedInputGate = moeSharedInputGate;
            this.moeRouter = moeRouter;
            this.isMoE = isMoE;
        }
    }
    
    public static final class State {
        public final FloatTensor x, xb, xb2, ssmTmp, ssmQkv, q, k, v, att, logits;
        public final FloatTensor[] keyCache, valueCache, ssmConvState, ssmState;
        public final FloatTensor ffnUp;
        public final FloatTensor moeRouterLogits, moeOutput, moeExpertOut, moeGateResult, moeUpResult;
        public final FloatTensor moeSharedGate, moeSharedUp, moeSharedOut, moeSharedInputGate;
        public final float[] ssmZ, ssmConvOut, ssmQ, ssmK, ssmV, ssmQGroup, ssmKGroup, ssmGate, ssmBeta, ssmOutput, ssmSk, ssmD;
        public final float[] attnGateArr, attnQArr, attnKArr;
        public final float[] ropeCr, ropeCi;
        public final int ropeHalf;
        public final int[] moeTopExperts;
        public final float[] moeTopWeights;
        public int latestToken;
        
        State(Configuration config) {
            int maxHiddenDim = Math.max(Math.max(config.embeddingLength, config.hiddenDim), 
                                        Math.max(config.ssmInnerSize, config.queryDim()));
            this.x = ArrayFloatTensor.allocate(config.embeddingLength);
            this.xb = ArrayFloatTensor.allocate(config.embeddingLength);
            this.xb2 = ArrayFloatTensor.allocate(maxHiddenDim);
            this.ssmTmp = ArrayFloatTensor.allocate(config.ssmInnerSize);
            this.ssmQkv = ArrayFloatTensor.allocate(config.convChannels());
            this.q = ArrayFloatTensor.allocate(2 * config.queryDim());
            this.k = ArrayFloatTensor.allocate(config.kvDim());
            this.v = ArrayFloatTensor.allocate(config.kvDim());
            this.att = ArrayFloatTensor.allocate(config.numberOfHeads, config.contextLength);
            this.logits = ArrayFloatTensor.allocate(config.vocabularySize);
            this.ffnUp = ArrayFloatTensor.allocate(maxHiddenDim);
            this.keyCache = new FloatTensor[config.numberOfLayers];
            this.valueCache = new FloatTensor[config.numberOfLayers];
            this.ssmConvState = new FloatTensor[config.numberOfLayers];
            this.ssmState = new FloatTensor[config.numberOfLayers];
            int convCh = config.convChannels();
            int headVDim = config.ssmInnerSize / config.ssmTimeStepRank;
            this.ssmZ = new float[config.ssmInnerSize];
            this.ssmConvOut = new float[convCh];
            this.ssmQ = new float[config.ssmInnerSize];
            this.ssmK = new float[config.ssmInnerSize];
            this.ssmV = new float[config.ssmInnerSize];
            this.ssmQGroup = new float[config.ssmGroupCount * headVDim];
            this.ssmKGroup = new float[config.ssmGroupCount * headVDim];
            this.ssmGate = new float[config.ssmTimeStepRank];
            this.ssmBeta = new float[config.ssmTimeStepRank];
            this.ssmOutput = new float[config.ssmInnerSize];
            this.ssmSk = new float[headVDim];
            this.ssmD = new float[headVDim];
            this.attnGateArr = new float[config.queryDim()];
            this.attnQArr = new float[config.queryDim()];
            this.attnKArr = new float[config.kvDim()];
            int ropeDim = Math.max(0, Math.min(config.ropeDimensionCount, config.headSize) & ~1);
            this.ropeHalf = ropeDim / 2;
            if (ropeHalf > 0) {
                Pair<float[], float[]> ropeFreqs = precomputeFreqsCis(config.contextLength, ropeDim, config.ropeTheta);
                this.ropeCr = ropeFreqs.first();
                this.ropeCi = ropeFreqs.second();
            } else {
                this.ropeCr = null;
                this.ropeCi = null;
            }
            if (config.isMoE()) {
                this.moeRouterLogits = ArrayFloatTensor.allocate(config.expertCount);
                this.moeOutput = ArrayFloatTensor.allocate(config.embeddingLength);
                this.moeExpertOut = ArrayFloatTensor.allocate(config.embeddingLength);
                this.moeGateResult = ArrayFloatTensor.allocate(Math.max(1, config.expertFeedForwardLength));
                this.moeUpResult = ArrayFloatTensor.allocate(Math.max(1, config.expertFeedForwardLength));
                this.moeSharedGate = ArrayFloatTensor.allocate(Math.max(1, config.expertSharedFeedForwardLength));
                this.moeSharedUp = ArrayFloatTensor.allocate(Math.max(1, config.expertSharedFeedForwardLength));
                this.moeSharedOut = ArrayFloatTensor.allocate(config.embeddingLength);
                this.moeSharedInputGate = ArrayFloatTensor.allocate(1);
                this.moeTopExperts = new int[Math.max(1, config.expertUsedCount)];
                this.moeTopWeights = new float[Math.max(1, config.expertUsedCount)];
            } else {
                this.moeRouterLogits = null;
                this.moeOutput = null;
                this.moeExpertOut = null;
                this.moeGateResult = null;
                this.moeUpResult = null;
                this.moeSharedGate = null;
                this.moeSharedUp = null;
                this.moeSharedOut = null;
                this.moeSharedInputGate = null;
                this.moeTopExperts = null;
                this.moeTopWeights = null;
            }
            for (int l = 0; l < config.numberOfLayers; l++) {
                if (config.isFullAttention[l]) {
                    keyCache[l] = ArrayFloatTensor.allocate(config.contextLength, config.kvDim());
                    valueCache[l] = ArrayFloatTensor.allocate(config.contextLength, config.kvDim());
                } else {
                    ssmConvState[l] = ArrayFloatTensor.allocate(config.ssmConvKernel - 1, convCh);
                    ssmState[l] = ArrayFloatTensor.allocate(headVDim, headVDim, config.ssmTimeStepRank);
                }
            }
        }
    }
    
    static void rmsnorm(FloatTensor out, FloatTensor x, FloatBuffer weight, int size, float rmsNormEps) {
        float ss = x.reduce(0, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        final float finalss = ss;
        out.mapWithIndexInPlace(0, size, (value, index) -> weight.get(index) * (finalss * x.getFloat(index)));
    }
    
    static void rmsnorm(FloatTensor out, int outOffset, FloatTensor x, int xOffset, FloatBuffer weight, int size, float rmsNormEps) {
        float ss = 0f;
        for (int i = 0; i < size; i++) {
            float xi = x.getFloat(xOffset + i);
            ss += xi * xi;
        }
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        for (int i = 0; i < size; i++) {
            out.setFloat(outOffset + i, weight.get(i) * ss * x.getFloat(xOffset + i));
        }
    }
    
    static Pair<float[], float[]> precomputeFreqsCis(int contextLength, int nRot, double theta) {
        int half = nRot / 2;
        float[] cr = new float[contextLength * half];
        float[] ci = new float[contextLength * half];
        for (int pos = 0; pos < contextLength; pos++) {
            for (int i = 0; i < half; i++) {
                double freq = 1.0 / Math.pow(theta, (2.0 * i) / nRot);
                double val = pos * freq;
                int n = pos * half + i;
                cr[n] = (float) Math.cos(val);
                ci[n] = (float) Math.sin(val);
            }
        }
        return new Pair<>(cr, ci);
    }

    static void applyMRoPE(float[] q, int headOffset, int position, float[] ropeCr, float[] ropeCi, int ropeHalf) {
        int ropeOffset = position * ropeHalf;
        for (int i = 0; i < ropeHalf; i++) {
            float fcr = ropeCr[ropeOffset + i];
            float fci = ropeCi[ropeOffset + i];
            int idx = headOffset + 2 * i;
            float v0 = q[idx];
            float v1 = q[idx + 1];
            q[idx] = v0 * fcr - v1 * fci;
            q[idx + 1] = v0 * fci + v1 * fcr;
        }
    }

    static float softplus(float x) {
        if (x > 20f) return x;
        if (x < -20f) return (float) Math.exp(x);
        return (float) Math.log1p(Math.exp(x));
    }

    static float sigmoid(float x) {
        return 1.0f / (1.0f + (float) Math.exp(-x));
    }

    static float silu(float x) {
        return x * sigmoid(x);
    }
    
    static void ssmForward(State state, Weights weights, Configuration config, int layer, int position) {
        // Qwen3.5 SSM parity checklist (llama.cpp qwen35 / delta-net base):
        // - attn_qkv projects to [d_inner + 2 * n_group * d_state] and then depthwise conv + SiLU.
        // - conv split order is [Q(n_group*d_state), K(n_group*d_state), V(dt_rank*head_v_dim)].
        // - Q/K are L2-normalized per group-head, then repeated to dt_rank heads when dt_rank != n_group.
        // - delta-net update uses exp(gate), beta sigmoid, and state layout [head_v_dim, head_v_dim, dt_rank].
        // - output path is RMSNorm(ssm_norm) gated by SiLU(z), then projected by ssm_out.
        int dim = config.embeddingLength;
        int dInner = config.ssmInnerSize;
        int nGroup = config.ssmGroupCount;
        int dtRank = config.ssmTimeStepRank;
        int dState = config.ssmStateSize;
        int convKernel = config.ssmConvKernel;
        int headVDim = dInner / dtRank;
        int convChannels = dInner + 2 * nGroup * dState;
        int qSize = dState * nGroup;
        int kOff = qSize;
        int vOff = 2 * qSize;
        final boolean parallelConv = convChannels >= 1024;
        final boolean parallelHeads = dtRank >= 16;
        final boolean parallelDelta = dtRank * headVDim * headVDim >= 8192;

        // 1. QKV projection: [convChannels] from [dim]
        FloatTensor qkv = state.ssmQkv;
        weights.attnQkv[layer].matmul(state.xb, qkv, convChannels, dim);

        // 2. z gating projection: [dInner] from [dim]
        float[] z = state.ssmZ;
        weights.attnGate[layer].matmul(state.xb, state.ssmTmp, dInner, dim);
        for (int i = 0; i < dInner; i++) z[i] = state.ssmTmp.getFloat(i);

        // 3. Causal 1D convolution with cached history.
        FloatTensor convState = state.ssmConvState[layer];
        FloatBuffer convWeight = weights.ssmConv1d[layer];
        float[] convOut = state.ssmConvOut;
        if (parallelConv) {
            Parallel.parallelFor(0, convChannels, c -> {
                float sum = 0;
                int wOff = c * convKernel;
                for (int k = 0; k < convKernel - 1; k++) {
                    sum += convWeight.get(wOff + k) * convState.getFloat(k * convChannels + c);
                }
                sum += convWeight.get(wOff + (convKernel - 1)) * qkv.getFloat(c);
                convOut[c] = sum;
            });
        } else {
            for (int c = 0; c < convChannels; c++) {
                float sum = 0;
                int wOff = c * convKernel;
                for (int k = 0; k < convKernel - 1; k++) {
                    sum += convWeight.get(wOff + k) * convState.getFloat(k * convChannels + c);
                }
                sum += convWeight.get(wOff + (convKernel - 1)) * qkv.getFloat(c);
                convOut[c] = sum;
            }
        }

        // Update conv cache: shift left, store current qkv as newest entry
        if (parallelConv) {
            Parallel.parallelFor(0, convChannels, c -> {
                for (int k = 0; k < convKernel - 2; k++) {
                    convState.setFloat(k * convChannels + c, convState.getFloat((k + 1) * convChannels + c));
                }
                convState.setFloat((convKernel - 2) * convChannels + c, qkv.getFloat(c));
            });
        } else {
            for (int k = 0; k < convKernel - 2; k++) {
                for (int c = 0; c < convChannels; c++) {
                    convState.setFloat(k * convChannels + c, convState.getFloat((k + 1) * convChannels + c));
                }
            }
            for (int c = 0; c < convChannels; c++) {
                convState.setFloat((convKernel - 2) * convChannels + c, qkv.getFloat(c));
            }
        }

        // 4. SiLU activation on conv output.
        if (parallelConv) {
            Parallel.parallelFor(0, convChannels, i -> convOut[i] = silu(convOut[i]));
        } else {
            for (int i = 0; i < convChannels; i++) {
                float v = convOut[i];
                convOut[i] = silu(v);
            }
        }

        // 5. Split conv output into Q, K, V and L2-normalize Q, K per head.
        if (dtRank % nGroup != 0) {
            throw new IllegalArgumentException("Unsupported SSM head config: dtRank=" + dtRank + ", nGroup=" + nGroup);
        }
        float scale = (float) (1.0 / Math.sqrt(headVDim));
        float[] qArr = state.ssmQ;
        float[] kArr = state.ssmK;
        float[] vArr = state.ssmV;
        float[] qGroup = state.ssmQGroup;
        float[] kGroup = state.ssmKGroup;

        if (parallelHeads && nGroup >= 8) {
            Parallel.parallelFor(0, nGroup, h -> {
                float qNormSq = 0, kNormSq = 0;
                int hOff = h * headVDim;
                int kHeadOff = kOff + hOff;
                for (int d = 0; d < headVDim; d++) {
                    float qv = convOut[hOff + d];
                    float kv = convOut[kHeadOff + d];
                    qNormSq += qv * qv;
                    kNormSq += kv * kv;
                }
                float qInvNorm = (float) (1.0 / Math.sqrt(qNormSq + config.rmsNormEps)) * scale;
                float kInvNorm = (float) (1.0 / Math.sqrt(kNormSq + config.rmsNormEps));
                for (int d = 0; d < headVDim; d++) {
                    qGroup[hOff + d] = convOut[hOff + d] * qInvNorm;
                    kGroup[hOff + d] = convOut[kHeadOff + d] * kInvNorm;
                }
            });
        } else {
            for (int h = 0; h < nGroup; h++) {
                float qNormSq = 0, kNormSq = 0;
                for (int d = 0; d < headVDim; d++) {
                    float qv = convOut[h * headVDim + d];
                    float kv = convOut[kOff + h * headVDim + d];
                    qNormSq += qv * qv;
                    kNormSq += kv * kv;
                }
                float qInvNorm = (float) (1.0 / Math.sqrt(qNormSq + config.rmsNormEps)) * scale;
                float kInvNorm = (float) (1.0 / Math.sqrt(kNormSq + config.rmsNormEps));
                for (int d = 0; d < headVDim; d++) {
                    qGroup[h * headVDim + d] = convOut[h * headVDim + d] * qInvNorm;
                    kGroup[h * headVDim + d] = convOut[kOff + h * headVDim + d] * kInvNorm;
                }
            }
        }

        // Repeat/tile Q,K heads to dtRank heads.
        if (parallelHeads) {
            Parallel.parallelFor(0, dtRank, h -> {
                int srcHead = h % nGroup;
                int dstOff = h * headVDim;
                int srcOff = srcHead * headVDim;
                for (int d = 0; d < headVDim; d++) {
                    qArr[dstOff + d] = qGroup[srcOff + d];
                    kArr[dstOff + d] = kGroup[srcOff + d];
                }
            });
        } else {
            for (int h = 0; h < dtRank; h++) {
                int srcHead = h % nGroup;
                int dstOff = h * headVDim;
                int srcOff = srcHead * headVDim;
                for (int d = 0; d < headVDim; d++) {
                    qArr[dstOff + d] = qGroup[srcOff + d];
                    kArr[dstOff + d] = kGroup[srcOff + d];
                }
            }
        }

        if (parallelHeads) {
            Parallel.parallelFor(0, dtRank, h -> {
                int off = h * headVDim;
                int src = vOff + off;
                for (int d = 0; d < headVDim; d++) {
                    vArr[off + d] = convOut[src + d];
                }
            });
        } else {
            for (int h = 0; h < dtRank; h++) {
                for (int d = 0; d < headVDim; d++) {
                    vArr[h * headVDim + d] = convOut[vOff + h * headVDim + d];
                }
            }
        }

        // 6. Compute gate = softplus(alpha @ xb + dt_bias) * A, beta = sigmoid(beta @ xb)
        weights.ssmAlpha[layer].matmul(state.xb, state.ssmTmp, dtRank, dim);
        float[] gate = state.ssmGate;
        if (parallelHeads) {
            Parallel.parallelFor(0, dtRank, h -> {
                float val = state.ssmTmp.getFloat(h) + weights.ssmDtBias[layer].get(h);
                val = softplus(val);
                gate[h] = val * weights.ssmA[layer].get(h);
            });
        } else {
            for (int h = 0; h < dtRank; h++) {
                float val = state.ssmTmp.getFloat(h) + weights.ssmDtBias[layer].get(h);
                val = softplus(val);
                gate[h] = val * weights.ssmA[layer].get(h);
            }
        }

        weights.ssmBeta[layer].matmul(state.xb, state.ssmTmp, dtRank, dim);
        float[] beta = state.ssmBeta;
        if (parallelHeads) {
            Parallel.parallelFor(0, dtRank, h -> beta[h] = sigmoid(state.ssmTmp.getFloat(h)));
        } else {
            for (int h = 0; h < dtRank; h++) {
                beta[h] = sigmoid(state.ssmTmp.getFloat(h));
            }
        }

        // 7. Gated delta net: autoregressive state update per head
        //    state layout: [headVDim, headVDim, numVHeads] — element (i, j, h) at offset h*HV^2 + j*HV + i
        float[] output = state.ssmOutput;
        FloatTensor ssmState = state.ssmState[layer];

        if (parallelDelta) {
            Parallel.parallelFor(0, dtRank, h -> {
                float expGate = (float) Math.exp(gate[h]);
                float betaH = beta[h];
                int stateBase = h * headVDim * headVDim;
                int headOff = h * headVDim;
                float[] sk = new float[headVDim];
                float[] d = new float[headVDim];

                for (int idx = 0; idx < headVDim * headVDim; idx++) {
                    int si = stateBase + idx;
                    ssmState.setFloat(si, ssmState.getFloat(si) * expGate);
                }

                for (int j = 0; j < headVDim; j++) {
                    float sum = 0;
                    for (int i = 0; i < headVDim; i++) {
                        sum += ssmState.getFloat(stateBase + j * headVDim + i) * kArr[headOff + i];
                    }
                    sk[j] = sum;
                }

                for (int i = 0; i < headVDim; i++) {
                    d[i] = (vArr[headOff + i] - sk[i]) * betaH;
                }

                for (int i = 0; i < headVDim; i++) {
                    float ki = kArr[headOff + i];
                    for (int j = 0; j < headVDim; j++) {
                        int si = stateBase + j * headVDim + i;
                        ssmState.setFloat(si, ssmState.getFloat(si) + ki * d[j]);
                    }
                }

                for (int j = 0; j < headVDim; j++) {
                    float sum = 0;
                    for (int i = 0; i < headVDim; i++) {
                        sum += ssmState.getFloat(stateBase + j * headVDim + i) * qArr[headOff + i];
                    }
                    output[headOff + j] = sum;
                }
            });
        } else {
            for (int h = 0; h < dtRank; h++) {
                float expGate = (float) Math.exp(gate[h]);
                float betaH = beta[h];
                int stateBase = h * headVDim * headVDim;
                int headOff = h * headVDim;

                // 7.1 Decay state: state *= exp(gate)
                for (int idx = 0; idx < headVDim * headVDim; idx++) {
                    int si = stateBase + idx;
                    ssmState.setFloat(si, ssmState.getFloat(si) * expGate);
                }

                // 7.2 sk = state^T @ k: sk[j] = Σ_i state[i][j] * k[i]
                float[] sk = state.ssmSk;
                for (int j = 0; j < headVDim; j++) {
                    float sum = 0;
                    for (int i = 0; i < headVDim; i++) {
                        sum += ssmState.getFloat(stateBase + j * headVDim + i) * kArr[headOff + i];
                    }
                    sk[j] = sum;
                }

                // 7.3 d = (v - sk) * beta
                float[] d = state.ssmD;
                for (int i = 0; i < headVDim; i++) {
                    d[i] = (vArr[headOff + i] - sk[i]) * betaH;
                }

                // 7.4 state += outer(k, d): state[i][j] += k[i] * d[j]
                for (int i = 0; i < headVDim; i++) {
                    float ki = kArr[headOff + i];
                    for (int j = 0; j < headVDim; j++) {
                        int si = stateBase + j * headVDim + i;
                        ssmState.setFloat(si, ssmState.getFloat(si) + ki * d[j]);
                    }
                }

                // 7.5 output = state^T @ q: o[j] = Σ_i state[i][j] * q[i]
                for (int j = 0; j < headVDim; j++) {
                    float sum = 0;
                    for (int i = 0; i < headVDim; i++) {
                        sum += ssmState.getFloat(stateBase + j * headVDim + i) * qArr[headOff + i];
                    }
                    output[headOff + j] = sum;
                }
            }
        }

        // 8. Gated normalization: RMSNorm(output) * SiLU(z), per head
        if (parallelHeads) {
            Parallel.parallelFor(0, dtRank, h -> {
                int headOff = h * headVDim;
                float ss = 0;
                for (int d = 0; d < headVDim; d++) {
                    float val = output[headOff + d];
                    ss += val * val;
                }
                float invRms = (float) (1.0 / Math.sqrt(ss / headVDim + config.rmsNormEps));
                for (int d = 0; d < headVDim; d++) {
                    float normed = output[headOff + d] * invRms * weights.ssmNorm[layer].get(d);
                    float zVal = z[headOff + d];
                    float siluZ = silu(zVal);
                    state.ssmTmp.setFloat(headOff + d, normed * siluZ);
                }
            });
        } else {
            for (int h = 0; h < dtRank; h++) {
                int headOff = h * headVDim;
                float ss = 0;
                for (int d = 0; d < headVDim; d++) {
                    float val = output[headOff + d];
                    ss += val * val;
                }
                float invRms = (float) (1.0 / Math.sqrt(ss / headVDim + config.rmsNormEps));
                for (int d = 0; d < headVDim; d++) {
                    float normed = output[headOff + d] * invRms * weights.ssmNorm[layer].get(d);
                    float zVal = z[headOff + d];
                    float siluZ = silu(zVal);
                    state.ssmTmp.setFloat(headOff + d, normed * siluZ);
                }
            }
        }

        // 9. Output projection: [dim] from [dInner]
        weights.ssmOut[layer].matmul(state.ssmTmp, state.xb, dim, dInner);
    }
    
    static void moeForward(State state, Weights weights, Configuration config, int layer) {
        int dim = config.embeddingLength;
        int expertFFN = config.expertFeedForwardLength;
        int numExperts = config.expertCount;
        int topK = Math.min(config.expertUsedCount, numExperts);
        
        // 1. Router projection: [numExperts] from [dim]
        FloatTensor routerLogits = state.moeRouterLogits;
        weights.moeRouter[layer].matmul(state.xb, routerLogits, numExperts, dim);
        
        // 2. Routing probabilities.
        routerLogits.softmaxInPlace(0, numExperts);
        
        // 3. Select top-K experts without sorting/boxing.
        int[] topExperts = state.moeTopExperts;
        float[] topWeights = state.moeTopWeights;
        for (int i = 0; i < topK; i++) {
            topExperts[i] = -1;
            topWeights[i] = Float.NEGATIVE_INFINITY;
        }
        for (int expertIdx = 0; expertIdx < numExperts; expertIdx++) {
            float prob = routerLogits.getFloat(expertIdx);
            int insertPos = -1;
            for (int k = 0; k < topK; k++) {
                if (prob > topWeights[k]) {
                    insertPos = k;
                    break;
                }
            }
            if (insertPos >= 0) {
                for (int k = topK - 1; k > insertPos; k--) {
                    topWeights[k] = topWeights[k - 1];
                    topExperts[k] = topExperts[k - 1];
                }
                topWeights[insertPos] = prob;
                topExperts[insertPos] = expertIdx;
            }
        }
        
        // 4. Normalize top-K probabilities
        float topKSum = 0f;
        for (int i = 0; i < topK; i++) {
            topKSum += topWeights[i];
        }
        float invTopK = topKSum == 0f ? 0f : 1f / topKSum;
        for (int i = 0; i < topK; i++) {
            topWeights[i] *= invTopK;
        }
        
        // 5. Aggregate expert outputs weighted by routing probabilities
        FloatTensor moeOutput = state.moeOutput;
        FloatTensor expertOut = state.moeExpertOut;
        FloatTensor gateResult = state.moeGateResult;
        FloatTensor upResult = state.moeUpResult;
        moeOutput.fillInPlace(0, dim, 0f);
        int gateUpStride = expertFFN * dim;
        int downStride = dim * expertFFN;
        
        for (int k = 0; k < topK; k++) {
            int expertIdx = topExperts[k];
            float weight = topWeights[k];
            if (expertIdx < 0 || weight <= 0f) continue;
            
            // Compute gate and up for this expert: SwiGLU
            int gateUpOffset = expertIdx * gateUpStride;
            int downOffset = expertIdx * downStride;
            weights.moeExpertGate[layer].matmul(state.xb, gateResult, expertFFN, dim, gateUpOffset);
            weights.moeExpertUp[layer].matmul(state.xb, upResult, expertFFN, dim, gateUpOffset);
            
            // SiLU activation on gate
            for (int i = 0; i < expertFFN; i++) {
                float val = gateResult.getFloat(i);
                gateResult.setFloat(i, silu(val));
            }
            
            // Element-wise multiply: gate * up
            gateResult.multiplyInPlace(0, upResult, 0, expertFFN);
            
            // Down projection: [dim] from [expertFFN]
            weights.moeExpertDown[layer].matmul(gateResult, expertOut, dim, expertFFN, downOffset);
            
            // Accumulate weighted output
            for (int i = 0; i < dim; i++) {
                moeOutput.setFloat(i, moeOutput.getFloat(i) + weight * expertOut.getFloat(i));
            }
        }
        
        // 6. Add shared expert if present
        if (config.expertSharedFeedForwardLength > 0 && weights.moeSharedGate[layer] != null) {
            int sharedFFN = config.expertSharedFeedForwardLength;
            FloatTensor sharedGate = state.moeSharedGate;
            FloatTensor sharedUp = state.moeSharedUp;
            FloatTensor sharedOut = state.moeSharedOut;
            FloatTensor sharedGateInp = state.moeSharedInputGate;
            
            weights.moeSharedGate[layer].matmul(state.xb, sharedGate, sharedFFN, dim);
            weights.moeSharedUp[layer].matmul(state.xb, sharedUp, sharedFFN, dim);
            
            // SiLU activation on shared gate
            for (int i = 0; i < sharedFFN; i++) {
                float val = sharedGate.getFloat(i);
                sharedGate.setFloat(i, silu(val));
            }
            
            sharedGate.multiplyInPlace(0, sharedUp, 0, sharedFFN);
            weights.moeSharedDown[layer].matmul(sharedGate, sharedOut, dim, sharedFFN);

            float sharedScale = 1.0f;
            if (weights.moeSharedInputGate[layer] != null) {
                weights.moeSharedInputGate[layer].matmul(state.xb, sharedGateInp, 1, dim);
                float gate = sharedGateInp.getFloat(0);
                sharedScale = sigmoid(gate);
            }
            
            // Add shared expert output
            for (int i = 0; i < dim; i++) {
                moeOutput.setFloat(i, moeOutput.getFloat(i) + sharedScale * sharedOut.getFloat(i));
            }
        }
        
        // Copy result to state.xb
        moeOutput.copyTo(0, state.xb, 0, dim);
    }
    
    static FloatTensor forward(Qwen35 model, State state, int token, int position) {
        Configuration config = model.configuration;
        Weights weights = model.weights;
        int dim = config.embeddingLength;
        int headSize = config.headSize;
        int kvDim = config.kvDim();
        int queryDim = config.queryDim();
        int kvMul = config.numberOfHeads / config.numberOfKeyValueHeads;
        weights.tokenEmbeddingTable.copyTo(token * dim, state.x, 0, dim);
        
        for (int l = 0; l < config.numberOfLayers; l++) {
            boolean isFullAttn = config.isFullAttention[l];
            rmsnorm(state.xb, state.x, weights.attnNorm[l], dim, config.rmsNormEps);
            
            if (isFullAttn) {
                int qGateDim = 2 * queryDim;
                weights.attnQ[l].matmul(state.xb, state.q, qGateDim, dim);
                float[] gateArr = state.attnGateArr;
                for (int h = 0; h < config.numberOfHeads; h++) {
                    for (int d = 0; d < headSize; d++) {
                        gateArr[h * headSize + d] = state.q.getFloat(h * 2 * headSize + headSize + d);
                        state.q.setFloat(h * headSize + d, state.q.getFloat(h * 2 * headSize + d));
                    }
                }
                for (int h = 0; h < config.numberOfHeads; h++) {
                    rmsnorm(state.q, h * headSize, state.q, h * headSize, weights.attnQNorm[l], headSize, config.rmsNormEps);
                }
                weights.attnK[l].matmul(state.xb, state.k, kvDim, dim);
                weights.attnV[l].matmul(state.xb, state.v, kvDim, dim);
                for (int h = 0; h < config.numberOfKeyValueHeads; h++) {
                    rmsnorm(state.k, h * headSize, state.k, h * headSize, weights.attnKNorm[l], headSize, config.rmsNormEps);
                }
                float[] qArr = state.attnQArr;
                float[] kArr = state.attnKArr;
                for (int i = 0; i < queryDim; i++) qArr[i] = state.q.getFloat(i);
                for (int i = 0; i < kvDim; i++) kArr[i] = state.k.getFloat(i);
                if (state.ropeHalf > 0) {
                    for (int h = 0; h < config.numberOfHeads; h++) {
                        applyMRoPE(qArr, h * headSize, position, state.ropeCr, state.ropeCi, state.ropeHalf);
                    }
                    for (int h = 0; h < config.numberOfKeyValueHeads; h++) {
                        applyMRoPE(kArr, h * headSize, position, state.ropeCr, state.ropeCi, state.ropeHalf);
                    }
                }
                for (int i = 0; i < queryDim; i++) state.q.setFloat(i, qArr[i]);
                for (int i = 0; i < kvDim; i++) state.k.setFloat(i, kArr[i]);
                state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
                state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);
                int finalL = l;
                Parallel.parallelFor(0, config.numberOfHeads, h -> {
                    int qOffset = h * headSize;
                    int attOffset = h * config.contextLength;
                    for (int t = 0; t <= position; t++) {
                        int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
                        float score = state.q.dot(qOffset, state.keyCache[finalL], keyCacheOffset, headSize);
                        score /= (float) Math.sqrt(headSize);
                        state.att.setFloat(attOffset + t, score);
                    }
                    state.att.softmaxInPlace(attOffset, position + 1);
                    int xbOffset = h * headSize;
                    state.xb2.fillInPlace(xbOffset, headSize, 0f);
                    for (int t = 0; t <= position; t++) {
                        int vOffset = t * kvDim + (h / kvMul) * headSize;
                        float a = state.att.getFloat(attOffset + t);
                        state.xb2.saxpyInPlace(xbOffset, state.valueCache[finalL], vOffset, headSize, a);
                    }
                });
                for (int i = 0; i < queryDim; i++) {
                    float g = gateArr[i];
                    float sigmoidG = sigmoid(g);
                    state.xb2.setFloat(i, state.xb2.getFloat(i) * sigmoidG);
                }
                weights.attnOutput[l].matmul(state.xb2, state.xb, dim, queryDim);
            } else {
                ssmForward(state, weights, config, l, position);
            }
            // Attention residual: cur = attn_out + x (inpSA)
            state.xb.addInPlace(0, state.x, 0, dim);
            // Save for FFN residual
            state.xb.copyTo(0, state.x, 0, dim);
            // Post-attention norm (applied AFTER residual, before FFN)
            rmsnorm(state.xb, state.xb, weights.postAttentionNorm[l], dim, config.rmsNormEps);
            
            // FFN: Use MoE or dense FFN based on configuration
            if (config.isMoE()) {
                moeForward(state, weights, config, l);
            } else {
                // Dense FFN: SwiGLU
                weights.ffnGate[l].matmul(state.xb, state.xb2, config.hiddenDim, dim);
                for (int i = 0; i < config.hiddenDim; i++) {
                    float x_val = state.xb2.getFloat(i);
                    state.xb2.setFloat(i, silu(x_val));
                }
                FloatTensor up = state.ffnUp;
                weights.ffnUp[l].matmul(state.xb, up, config.hiddenDim, dim);
                state.xb2.multiplyInPlace(0, up, 0, config.hiddenDim);
                weights.ffnDown[l].matmul(state.xb2, state.xb, dim, config.hiddenDim);
            }
            // FFN residual: x = ffn_out + (attn_out + inpSA)
            state.x.addInPlace(state.xb);
        }
        rmsnorm(state.x, state.x, weights.outputNorm, dim, config.rmsNormEps);
        weights.outputWeight.matmul(state.x, state.logits, config.vocabularySize, dim);
        return state.logits;
    }
    
    private static final String ANSI_CYAN = "\033[36m";
    private static final String ANSI_RESET = "\033[0m";

    public static List<Integer> generateTokens(Qwen35 model, State state, int startPosition, List<Integer> promptTokens,
                                               Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
                                               boolean color, IntConsumer onTokenGenerated) {
        long startNanos = System.nanoTime(), startGen = 0;
        if (maxTokens < 0 || model.configuration.contextLength < maxTokens) maxTokens = model.configuration.contextLength;
        List<Integer> generatedTokens = new ArrayList<>(maxTokens);
        int promptIndex = 0;
        int token = state.latestToken;
        if (token < 0 && !promptTokens.isEmpty()) {
            token = promptTokens.get(0);
            promptIndex = 1;
        }
        int nextToken;
        
        for (int position = startPosition; position < maxTokens; ++position) {
            forward(model, state, token, position);
            if (promptIndex < promptTokens.size()) {
                nextToken = promptTokens.get(promptIndex++);
                if (echo) System.err.print(QwenTokenizer.replaceControlCharacters(model.tokenizer.decode(List.of(nextToken))));
                if (promptIndex >= promptTokens.size()) startGen = System.nanoTime();
            } else {
                nextToken = sampler.sampleToken(state.logits);
                if (echo) System.err.print(QwenTokenizer.replaceControlCharacters(model.tokenizer.decode(List.of(nextToken))));
                generatedTokens.add(nextToken);
                if (onTokenGenerated != null) onTokenGenerated.accept(nextToken);
                if (stopTokens.contains(nextToken)) break;
            }
            state.latestToken = token = nextToken;
        }
        long elapsedNanos = System.nanoTime() - startNanos;
        long promptNanos = startGen - startNanos;
        long genNanos = elapsedNanos - startGen + startNanos;
        String timingPrefix = color ? ANSI_CYAN : "";
        String timingSuffix = color ? ANSI_RESET : "";
        System.err.printf("%n%scontext: %d/%d prompt: %.2f tokens/s (%d) generation: %.2f tokens/s (%d)%s%n",
                timingPrefix,
                startPosition + promptIndex + generatedTokens.size(), model.configuration.contextLength,
                promptTokens.size() / (promptNanos / 1_000_000_000.0), promptTokens.size(),
                generatedTokens.size() / (genNanos / 1_000_000_000.0), generatedTokens.size(),
                timingSuffix);
        return generatedTokens;
    }
    
    
    private static final String ANSI_GREY  = "\033[90m";
    // ANSI_RESET already defined above

    /**
     * Buffered streaming decoder that accumulates BPE token bytes and flushes
     * only when complete UTF-8 characters are available. This prevents partial
     * multi-byte characters (e.g. emojis) from being printed as garbage.
     */
    static final class StreamingDecoder {
        private final Vocabulary vocabulary;
        private final byte[] buffer = new byte[16]; // enough for any UTF-8 sequence
        private int buffered = 0;

        StreamingDecoder(Vocabulary vocabulary) {
            this.vocabulary = vocabulary;
        }

        String decode(int token) {
            String tokenStr = vocabulary.get(token);
            // Convert BPE token string through byte decoder
            int[] codePoints = tokenStr.codePoints().toArray();
            for (int cp : codePoints) {
                Integer byteVal = QwenTokenizer.BYTE_DECODER.get(cp);
                if (byteVal == null) {
                    // Not a BPE byte char (shouldn't happen for non-special tokens)
                    continue;
                }
                buffer[buffered++] = (byte) byteVal.intValue();
            }
            // Try to decode as much valid UTF-8 as possible
            String result = flushValid();
            return result;
        }

        private String flushValid() {
            if (buffered == 0) return "";
            // Find how many bytes form complete UTF-8 characters
            int validEnd = 0;
            int i = 0;
            while (i < buffered) {
                int b = buffer[i] & 0xFF;
                int seqLen;
                if (b < 0x80) seqLen = 1;
                else if ((b & 0xE0) == 0xC0) seqLen = 2;
                else if ((b & 0xF0) == 0xE0) seqLen = 3;
                else if ((b & 0xF8) == 0xF0) seqLen = 4;
                else { i++; validEnd = i; continue; } // invalid lead byte, skip
                if (i + seqLen <= buffered) {
                    validEnd = i + seqLen;
                    i += seqLen;
                } else {
                    break; // incomplete sequence, keep in buffer
                }
            }
            if (validEnd == 0) return "";
            String result = new String(buffer, 0, validEnd, StandardCharsets.UTF_8);
            // Shift remaining bytes
            int remaining = buffered - validEnd;
            if (remaining > 0) {
                System.arraycopy(buffer, validEnd, buffer, 0, remaining);
            }
            buffered = remaining;
            return result;
        }
    }

    private static void onThinkingStart(PrintStream thoughtOut, boolean ansi) {
        if (ansi) {
            thoughtOut.print(ANSI_GREY);
        }
        thoughtOut.println("[Start thinking]");
    }

    private static void onThinkingEnd(PrintStream thoughtOut, boolean ansi, boolean emitted) {
        if (emitted) {
            thoughtOut.println();
        }
        thoughtOut.println("[End thinking]");
        if (ansi) {
            thoughtOut.print(ANSI_RESET);
        }
        thoughtOut.println();
    }

    static boolean supportsAnsiColors(String colorMode) {
        return switch (colorMode) {
            case "on" -> true;
            case "off" -> false;
            case "auto" -> {
                if (System.console() == null) {
                    yield false;
                }
                String noColor = System.getenv("NO_COLOR");
                if (noColor != null) {
                    yield false;
                }
                String term = System.getenv("TERM");
                yield term == null || !"dumb".equalsIgnoreCase(term);
            }
            default -> false;
        };
    }

    private static IntConsumer streamingPrinter(QwenTokenizer tokenizer, Options options) {
        if (!options.stream()) {
            return token -> {};
        }

        Integer thinkOpen = tokenizer.specialTokens().get("<think>");
        Integer thinkClose = tokenizer.specialTokens().get("</think>");
        StreamingDecoder decoder = new StreamingDecoder(tokenizer.vocabulary());

        boolean thinkEnabled = options.think();
        PrintStream thoughtOut = options.thinkInline() ? System.out : System.err;
        boolean ansi = options.colors();
        boolean[] inThink = {false};
        boolean[] emitted = {false};
        return token -> {
            if (thinkOpen != null && token == thinkOpen) {
                if (thinkEnabled) {
                    onThinkingStart(thoughtOut, ansi);
                }
                inThink[0] = true;
                emitted[0] = false;
                return;
            }
            if (thinkClose != null && token == thinkClose) {
                if (thinkEnabled) {
                    onThinkingEnd(thoughtOut, ansi, emitted[0]);
                }
                inThink[0] = false;
                emitted[0] = false;
                return;
            }
            if (!tokenizer.isSpecialToken(token)) {
                String text = decoder.decode(token);
                if (!text.isEmpty()) {
                    if (inThink[0]) {
                        if (thinkEnabled) {
                            thoughtOut.print(text);
                            emitted[0] = true;
                        }
                    } else {
                        System.out.print(text);
                    }
                }
            }
        };
    }

    private static List<Integer> visibleTokens(QwenTokenizer tokenizer, List<Integer> tokens, boolean think) {
        return think ? stripThinkTokens(tokenizer, tokens) : tokens;
    }

    private static List<Integer> stripThinkTokens(QwenTokenizer tokenizer, List<Integer> tokens) {
        Integer thinkOpen = tokenizer.specialTokens().get("<think>");
        Integer thinkClose = tokenizer.specialTokens().get("</think>");
        if (thinkOpen == null || thinkClose == null || tokens.isEmpty()) {
            return tokens;
        }
        List<Integer> out = new ArrayList<>(tokens.size());
        boolean inThink = false;
        for (int tok : tokens) {
            if (tok == thinkOpen) { inThink = true; continue; }
            if (tok == thinkClose) { inThink = false; continue; }
            if (!inThink) { out.add(tok); }
        }
        return out;
    }

    static final int DEFAULT_MAX_TOKENS = 1024;

    static Sampler selectSampler(int vocabularySize, float temperature, float topp, long rngSeed) {
        if (temperature == 0.0f) return Sampler.ARGMAX;
        RandomGenerator rng = RandomGeneratorFactory.getDefault().create(rngSeed);
        Sampler innerSampler = (topp <= 0 || topp >= 1) ? new CategoricalSampler(rng) : new ToppSampler(vocabularySize, topp, rng);
        return logits -> {
            int logitsSize = Math.toIntExact(logits.size());
            logits.divideInPlace(0, logitsSize, temperature);
            logits.softmaxInPlace(0, logitsSize);
            return innerSampler.sampleToken(logits);
        };
    }
    
    static void runInteractive(Qwen35 model, Sampler sampler, Options options) {
        Qwen35.State state = null;
        QwenChatFormat chatFormat = new QwenChatFormat(model.tokenizer);
        List<Integer> conversationTokens = new ArrayList<>();
        if (options.systemPrompt() != null) conversationTokens.addAll(chatFormat.encodeMessage("system", options.systemPrompt()));
        int startPosition = 0;
        Scanner in = new Scanner(System.in);
        loop: while (true) {
            System.out.print("> ");
            System.out.flush();
            if (!in.hasNextLine()) {
                break;
            }
            String userText = in.nextLine();
            switch (userText) {
                case "/quit":
                case "/exit": break loop;
                case "/context":
                    System.out.printf("%d out of %d context tokens used (%d tokens remaining)%n",
                            conversationTokens.size(), options.maxTokens(), options.maxTokens() - conversationTokens.size());
                    continue;
            }
            if (state == null) state = model.createNewState();
            conversationTokens.addAll(chatFormat.encodeMessage("user", userText));
            conversationTokens.addAll(chatFormat.encodeHeader("assistant", options.think()));
            Set<Integer> stopTokens = chatFormat.getStopTokens();
            IntConsumer printer = streamingPrinter(model.tokenizer, options);
            List<Integer> responseTokens = Qwen35.generateTokens(model, state, startPosition,
                    conversationTokens.subList(startPosition, conversationTokens.size()),
                    stopTokens, options.maxTokens(), sampler, options.echo(), options.colors(), printer);
            Integer stopToken = null;
            if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
                stopToken = responseTokens.getLast();
                responseTokens.removeLast();
            }
            List<Integer> visibleResponseTokens = visibleTokens(model.tokenizer, responseTokens, options.think());
            conversationTokens.addAll(responseTokens);
            if (stopToken != null) {
                conversationTokens.add(stopToken);
            }
            startPosition = conversationTokens.size();
            if (!options.stream()) {
                System.out.println(model.tokenizer.decode(visibleResponseTokens));
            }
            if (stopToken == null) {
                System.err.println("Ran out of context length...");
                break;
            }
        }
    }
    
    static void runInstructOnce(Qwen35 model, Sampler sampler, Options options) {
        Qwen35.State state = model.createNewState();
        QwenChatFormat chatFormat = new QwenChatFormat(model.tokenizer);
        List<Integer> promptTokens = new ArrayList<>();
        if (options.systemPrompt() != null) promptTokens.addAll(chatFormat.encodeMessage("system", options.systemPrompt()));
        promptTokens.addAll(chatFormat.encodeMessage("user", options.prompt()));
        promptTokens.addAll(chatFormat.encodeHeader("assistant", options.think()));
        Set<Integer> stopTokens = chatFormat.getStopTokens();
        IntConsumer printer = streamingPrinter(model.tokenizer, options);
        List<Integer> responseTokens = Qwen35.generateTokens(model, state, 0, promptTokens, stopTokens,
                options.maxTokens(), sampler, options.echo(), options.colors(), printer);
        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) responseTokens.removeLast();
        List<Integer> visibleResponseTokens = visibleTokens(model.tokenizer, responseTokens, options.think());
        if (!options.stream()) System.out.println(model.tokenizer.decode(visibleResponseTokens));
    }
    
    record Options(Path modelPath, String prompt, String systemPrompt, boolean interactive,
                   float temperature, float topp, long seed, int maxTokens, boolean stream, boolean echo,
                   boolean think, boolean thinkInline, boolean colors) {
        Options {
            require(modelPath != null, "Missing argument: --model <path> is required");
            require(interactive || prompt != null, "Missing argument: --prompt is required unless --chat mode");
            require(temperature >= 0, "Invalid argument: --temperature must be non-negative");
            require(topp >= 0 && topp <= 1, "Invalid argument: --top-p must be within [0, 1]");
        }

        static void require(boolean condition, String messageFormat, Object... args) {
            if (!condition) {
                System.out.println("ERROR " + messageFormat.formatted(args));
                System.out.println();
                printUsage(System.out);
                System.exit(-1);
            }
        }

        static boolean parseBooleanOption(String optionName, String value) {
            return switch (value.toLowerCase(Locale.ROOT)) {
                case "true", "on" -> true;
                case "false", "off" -> false;
                default -> {
                    require(false, "Invalid argument for %s: expected true|false|on|off, got %s", optionName, value);
                    yield false;
                }
            };
        }

        static void printUsage(PrintStream out) {
            out.println("Usage:  jbang Qwen35.java [options]");
            out.println();
            out.println("Options:");
            out.println("  --model, -m <path>            required, path to .gguf file");
            out.println("  --interactive, --chat, -i     run in chat mode");
            out.println("  --instruct                    run in instruct (once) mode, default mode");
            out.println("  --prompt, -p <string>         input prompt");
            out.println("  --system-prompt, -sp <string> system prompt");
            out.println("  --temperature, -temp <float>  temperature in [0,inf], default 0.7");
            out.println("  --top-p <float>               p value in top-p (nucleus) sampling in [0,1] default 0.95");
            out.println("  --seed <long>                 random seed, default System.nanoTime()");
            out.println("  --max-tokens, -n <int>        number of steps to run for < 0 = limited by context length, default " + DEFAULT_MAX_TOKENS);
            out.println("  --stream <boolean>            print tokens during generation; accepts true|false|on|off, default true");
            out.println("  --echo <boolean>              print ALL tokens to stderr; accepts true|false|on|off, default false");
            out.println("  --color <on|off|auto>         colorize thinking output in terminal (default: auto)");
            out.println("  --think <off|on|inline>       off: disable thoughts, on: thoughts to stderr, inline: thoughts to stdout");
            out.println();
            out.println("Interactive commands:");
            out.println("  /quit, /exit                  exit the chat");
            out.println("  /context                      show context token usage");
            out.println();
            out.println("Examples:");
            out.println("  jbang Qwen35.java --model Qwen3.5-35B-A3B-Q8_0.gguf --chat");
            out.println("  jbang Qwen35.java --model Qwen3.5-35B-A3B-Q8_0.gguf --prompt \"Tell me a joke\"");
            out.println("  jbang Qwen35.java --model Qwen3.5-35B-A3B-Q8_0.gguf --chat --system-prompt \"You are a helpful assistant\"");
        }

        static Options parseOptions(String[] args) {
            String prompt = null, systemPrompt = null;
            float temperature = 0.7f, topp = 0.95f;
            Path modelPath = null;
            long seed = System.nanoTime();
            int maxTokens = DEFAULT_MAX_TOKENS;
            boolean interactive = false, stream = true, echo = false;
            boolean think = false, thinkInline = false;
            String colorMode = "auto";

            for (int i = 0; i < args.length; i++) {
                String optionName = args[i];
                require(optionName.startsWith("-"), "Invalid option %s", optionName);
                switch (optionName) {
                    case "--interactive", "--chat", "-i" -> interactive = true;
                    case "--instruct" -> interactive = false;
                    case "--help", "-h" -> { printUsage(System.out); System.exit(0); }
                    default -> {
                        String nextArg;
                        if (optionName.contains("=")) {
                            String[] parts = optionName.split("=", 2);
                            optionName = parts[0];
                            nextArg = parts[1];
                        } else {
                            require(i + 1 < args.length, "Missing argument for option %s", optionName);
                            nextArg = args[i + 1];
                            i += 1;
                        }
                        switch (optionName) {
                            case "--prompt", "-p" -> prompt = nextArg;
                            case "--system-prompt", "-sp" -> systemPrompt = nextArg;
                            case "--temperature", "--temp" -> temperature = Float.parseFloat(nextArg);
                            case "--top-p" -> topp = Float.parseFloat(nextArg);
                            case "--model", "-m" -> modelPath = Path.of(nextArg);
                            case "--seed", "-s" -> seed = Long.parseLong(nextArg);
                            case "--max-tokens", "-n" -> maxTokens = Integer.parseInt(nextArg);
                            case "--stream" -> stream = parseBooleanOption(optionName, nextArg);
                            case "--echo" -> echo = parseBooleanOption(optionName, nextArg);
                            case "--color" -> colorMode = nextArg.toLowerCase(Locale.ROOT);
                            case "--think" -> {
                                String thinkMode = nextArg.toLowerCase(Locale.ROOT);
                                thinkInline = List.of("inline", "stdout").contains(thinkMode);
                                switch (thinkMode) {
                                    case "on", "true", "inline", "stdout" -> think = true;
                                    case "off", "false" -> think = false;
                                    default -> require(false, "Invalid argument for %s: expected off|on|inline (or false|true|stdout), got %s", optionName, nextArg);
                                }
                            }
                            default -> require(false, "Unknown option: %s", optionName);
                        }
                    }
                }
            }
            require(List.of("on", "off", "auto").contains(colorMode), "Invalid argument: --color must be one of on|off|auto");
            boolean color = Qwen35.supportsAnsiColors(colorMode);
            return new Options(modelPath, prompt, systemPrompt, interactive, temperature, topp, seed, maxTokens, stream, echo, think, thinkInline, color);
        }
    }
    
    public static void main(String[] args) throws IOException {
        Options options = Options.parseOptions(args);
        Qwen35 model = AOT.tryUsePreLoaded(options.modelPath(), options.maxTokens());
        if (model == null) {
            model = ModelLoader.loadModel(options.modelPath(), options.maxTokens());
        }
        Sampler sampler = selectSampler(model.configuration.vocabularySize, options.temperature(), options.topp(), options.seed());
        if (options.interactive()) runInteractive(model, sampler, options);
        else runInstructOnce(model, sampler, options);
    }
}

final class AOT {
    record PartialModel(String modelFileName, GGUF gguf) {}

    private static final PartialModel PRELOADED_GGUF = preLoadGGUF(System.getProperty("qwen35.PreloadGGUF"));

    private static PartialModel preLoadGGUF(String modelPath) {
        if (modelPath == null || modelPath.isEmpty()) {
            return null;
        }
        try {
            Path path = Path.of(modelPath);
            if (!Files.exists(path) || !Files.isRegularFile(path)) {
                throw new IllegalArgumentException("Cannot pre-load model: " + path);
            }
            try (FileChannel fileChannel = FileChannel.open(path, StandardOpenOption.READ)) {
                GGUF gguf = GGUF.loadModel(fileChannel, path.toString());
                return new PartialModel(path.getFileName().toString(), gguf);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static Qwen35 tryUsePreLoaded(Path modelPath, int contextLength) throws IOException {
        AOT.PartialModel preLoaded = AOT.PRELOADED_GGUF;
        if (preLoaded == null) {
            return null;
        }
        String optionsModel = modelPath.getFileName().toString();
        String preLoadedModel = preLoaded.modelFileName();
        if (!Objects.equals(optionsModel, preLoadedModel)) {
            return null;
        }
        try (var timer = Timer.log("Load tensors from pre-loaded model");
             var fileChannel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            return ModelLoader.loadModel(fileChannel, preLoaded.gguf(), contextLength);
        }
    }
}
