package kr.co.hanbit.imageclassifier2;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.text.InputType;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.FileDescriptor;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class ImageClassifier {
    private static final float[] IMAGE_MEAN_IMAGENET = new float[] {103.939f, 116.779f, 123.68f};
    static int INPUT_SIZE = 224;
    static  int PIXEL_SIZE = 3;

    static AssetManager assetManager;
    static String modelPath = "";
    static Interpreter interpreter;

    public static String predict(Bitmap image) {
        ByteBuffer inpBuffer = preProcessImage(image);
        Tensor outTensor = interpreter.getOutputTensor(0);
        int[] outShape = outTensor.shape();
        DataType outType = outTensor.dataType();
        Log.d("datatype is", "predict: "+ outType);
        float[][] out = new float[outShape[0]][outShape[1]];
        // Model inference implemented.
        interpreter.run(inpBuffer, out);
        Log.d("output is ", "predict: " + Arrays.toString(out[0]));
        float[][] out_float = new float[1][1];
        out_float[0][0] = 1.0f;
        if (Arrays.equals(out[0], out_float[0])) {
            // return "Dog";
            return "Grass";
        }
        else {
            // return "Cat";
            return "Dandelion";
        }
    }

    private static ByteBuffer preProcessImage(Bitmap bitmap) {
        bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i=0; i< INPUT_SIZE; i++) {
            for (int j=0; j<INPUT_SIZE; j++) {
                int input = intValues[pixel++];

                byteBuffer.putFloat(((float)(input>>16  & 0xFF) - IMAGE_MEAN_IMAGENET[2]));
                byteBuffer.putFloat(((float)(input>>8 & 0xFF) - IMAGE_MEAN_IMAGENET[1]));
                byteBuffer.putFloat(((float)(input & 0xFF) - IMAGE_MEAN_IMAGENET[0]));
            }
        }
        return byteBuffer;
    }

    public static void init(AssetManager assetManager, String model_path){
        ImageClassifier.assetManager = assetManager;
        ImageClassifier.modelPath = model_path;
        interpreter = createInterpreter(assetManager, model_path);
    }

    static Interpreter createInterpreter(AssetManager assetManager, String model_path){
        Interpreter.Options options= new Interpreter.Options();
        options.setNumThreads(5);
        options.setUseNNAPI(true);
        // Create interpreter and return.
        return new Interpreter(loadModelFile(assetManager, model_path), options);
    }

    private static ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) {
        AssetFileDescriptor fileDescriptor = null;
        try {
            fileDescriptor = assetManager.openFd(modelPath);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }
}
