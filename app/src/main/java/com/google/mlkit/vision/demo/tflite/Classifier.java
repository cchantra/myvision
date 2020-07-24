package com.google.mlkit.vision.demo.tflite;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Build;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.google.gson.JsonParser;
import com.google.mlkit.vision.demo.AsyncResponse;
import com.google.mlkit.vision.demo.Constant;
import com.google.mlkit.vision.demo.FaceAPIUtil;
import com.google.mlkit.vision.demo.Logger;
import com.google.mlkit.vision.demo.constantURL;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.lang.reflect.Array;
import java.net.CookieStore;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.MappedByteBuffer;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.GregorianCalendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;



public abstract  class Classifier {

    private static final Logger LOGGER = new Logger();



    /**
     * Number of results to show in the UI.
     */
    private static final int MAX_RESULTS = 3;


    /**
     * The loaded TensorFlow Lite model.
     */
    private MappedByteBuffer tfliteModel;

    /**
     * Image size along the x axis.
     */
    private int imageSizeX = 0;

    /**
     * Image size along the y axis.
     */
    private int imageSizeY = 0;

    /**
     * Optional GPU delegate for accleration.
     */
    private GpuDelegate gpuDelegate = null;

    /**
     * Optional NNAPI delegate for accleration.
     */
    private NnApiDelegate nnApiDelegate = null;

    /**
     * An instance of the driver class to run model inference with Tensorflow Lite.
     */
    protected Interpreter tflite;

    /**
     * Options for configuring the Interpreter.
     */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    /**
     * Labels corresponding to the output of the vision model.
     */
    private List<String> labels;

    /**
     * Input image TensorBuffer.
     */
    private TensorImage inputImageBuffer;

    /**
     * Output probability TensorBuffer.
     */
    private TensorBuffer outputProbabilityBuffer;

    /**
     * Processer to apply post processing of the output probability.
     */
    private  TensorProcessor probabilityProcessor;

    public static int currentModel;

    List<Recognition>  recogList;

    public static void   create(Activity activity, int mode)
              {
        currentModel = mode;

    }

    /** An immutable result returned by a Classifier describing what was recognized. */
    public static class Recognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final String id;

        /** Display name for the recognition. */
        private final String title;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private final Float confidence;

        /** Optional location within the source image for the location of the recognized object. */
        private RectF location;

        public Recognition(
                final String id, final String title, final Float confidence, final RectF location) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        public void setLocation(RectF location) {
            this.location = location;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }

    /** Initializes a {@code Classifier}. */
    protected Classifier(Activity activity ) throws IOException {
        if (currentModel == Constant.TEST_JSON) {

            System.out.println("FACE API");
        }
        else {

            tfliteModel = FileUtil.loadMappedFile(activity, getModelPath());

            //tfliteOptions.setNumThreads(numThreads);
            tflite = new Interpreter(tfliteModel, tfliteOptions);

            // Loads labels out from the label file.
            labels = FileUtil.loadLabels(activity, getLabelPath());

            // Reads type and shape of input and output tensors, respectively.
            int imageTensorIndex = 0;
            int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
            imageSizeY = imageShape[1];
            imageSizeX = imageShape[2];
            System.out.println("---" + imageShape[0] + " " + imageShape[1] + " " + imageShape[2] + " " + imageShape[3]);
            DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();
            int probabilityTensorIndex = 0;
            int[] probabilityShape =
                    tflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
            DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();

            // Creates the input tensor.
            inputImageBuffer = new TensorImage(imageDataType);

            // Creates the output tensor and its processor.
            outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

            // Creates the post processor for the output probability.
            probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();

            LOGGER.d("Created a Tensorflow Lite Image Classifier.");
        }
    }


    /** Runs inference and returns the classification results. */
    /** Runs inference and returns the classification results. */
    @RequiresApi(api = Build.VERSION_CODES.JELLY_BEAN_MR2)

    public List<Recognition> recognizeImage(final Bitmap bitmap ) throws MalformedURLException {
        // Logs this method so that it can be analyzed with systrace.

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN_MR2) {
            Trace.beginSection("recognizeImage");
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN_MR2) {
            Trace.beginSection("loadImage");
        }

        long startTimeForLoadImage = SystemClock.uptimeMillis();
        inputImageBuffer = loadImage(bitmap);
        long endTimeForLoadImage = SystemClock.uptimeMillis();
        Trace.endSection();
        LOGGER.v("Timecost to load the image: " + (endTimeForLoadImage - startTimeForLoadImage));

        // Runs the inference call.
        Trace.beginSection("runInference");
        long startTimeForReference = SystemClock.uptimeMillis();
        //System.out.println(inputImageBuffer.getHeight());
        tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
        long endTimeForReference = SystemClock.uptimeMillis();
        Trace.endSection();
        LOGGER.v("Timecost to run model inference: " + (endTimeForReference - startTimeForReference));

        // Gets the map of label and probability.
        Map<String, Float> labeledProbability =
                new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
                        .getMapWithFloatValue();
        Trace.endSection();

        // Gets top-k results.

                recogList = getTopKProbability(labeledProbability);

        return recogList;
    }


    public void recognizeImageAll(final Bitmap bitmap, List<List<Recognition>> recogListAll) throws MalformedURLException {
        // Logs this method so that it can be analyzed with systrace.

        //List<List<Recognition>> recogListAll = new ArrayList<List<Recognition>>();

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN_MR2) {
            Trace.beginSection("recognizeImage");
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN_MR2) {
            Trace.beginSection("loadImage");
        }
        if (currentModel == Constant.TEST_JSON) {
            Bitmap [] bma = {bitmap};

            FaceAPIUtil asyncTask = new FaceAPIUtil(new URL(constantURL.FaceURL), output -> {
                //Here you will receive the result fired from async class
                //of onPostExecute(result) method.

                Log.e("output", output);

                // parse : {'emotion': [['sad'], ['neutral']], 'detail': [[[1, 2, 3, 4], [0.56]], [[1, 2, 3, 5], [0.44]]]}
                JSONObject json = new JSONObject(output);
                JSONArray emotionArray = (JSONArray) json.get("emotion");
                JSONArray detailArray = (JSONArray) json.get("detail");
                Map<String, Object []> labeledProbability =  new HashMap<String, Object[]>();

                for (int i=0; i < emotionArray.length(); i++) { // for each face
                    System.out.println(emotionArray.get(i));
                    System.out.println(detailArray.get(i));


                    JSONArray plist = (JSONArray) emotionArray.get(i);
                    JSONArray dlist = (JSONArray) detailArray.get(i);

                    //String [] plist = (String []) obj; // plist[i] is array for each face
                    //Object [] dlist = (Object [] ) detailArray.get(i); // array for each face
                    for (int j=0; j < plist.length() ; j++) { // list of predictions
                        Log.e("class", (String) plist.get(j));

                        //Object [] ddlist = (Object[]) dlist[j];
                        JSONArray  loc_array = (JSONArray)dlist.get(0);
                        System.out.println(loc_array);

                        JSONArray confs = (JSONArray)  dlist.get(1);


                        System.out.println("list of confs");
                        System.out.println(confs);
                        Log.e("prob" ,   confs.get(j).toString());
                        // insert into hashMap
                        Object [] values = new Object[2];
                        values[0] = (double ) confs.get(j);
                        values[1] = new RectF( (int) loc_array.get(0), (int) loc_array.get(1),   (int)loc_array.get(0)+(int)loc_array.get(2), (int)loc_array.get(1)+ (int)loc_array.get(3));
                        labeledProbability.put((String) plist.get(j),values);


                    }
                }


                recogListAll.add(getTopKProbabilityWithLocation(labeledProbability));
                System.out.println("added "+recogListAll.size());

            });

            asyncTask.execute(bma);
        }

        //return recogListAll;
    }

    /** Closes the interpreter and model to release resources. */
    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
        
        if (nnApiDelegate != null) {
            nnApiDelegate.close();
            nnApiDelegate = null;
        }
        tfliteModel = null;
    }

    /** Get the image size along the x axis. */
    public int getImageSizeX() {
        return imageSizeX;
    }

    /** Get the image size along the y axis. */
    public int getImageSizeY() {
        return imageSizeY;
    }

    /** Loads input image, and applies preprocessing. */
    private TensorImage loadImage(final Bitmap bitmap ) {
        // Loads bitmap into a TensorImage.




        inputImageBuffer.load(bitmap);
        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
      //  int numRotation = sensorOrientation / 90;
        // TODO(b/143564309): Fuse ops inside ImageProcessor.

        ImageProcessor imageProcessor;
        if (currentModel == Constant.AGE_OPTION ||  currentModel == Constant.EXP_OPTION || currentModel == Constant.GENDER_OPTION) {
            System.out.println(currentModel);

            imageProcessor =
                    new ImageProcessor.Builder()
                            //  .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                            .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                           // .add(new Rot90Op(numRotation))
                            .add(getPreprocessNormalizeOp())
                            .build();
            return imageProcessor.process(inputImageBuffer);
        }
    /*else if (currentModel == Model.EMOTIONNET) {

      System.out.println("Image size"+ imageSizeX+ " " + imageSizeY);
     // Bitmap newImg = ImageUtils.resizeBitmap(bitmap, imageSizeX,imageSizeY);
      //newImg = ImageUtils.toGrayscale(newImg);
      //System.out.println(newImg.getWidth()+" "+newImg.getHeight());
      imageProcessor =
              new ImageProcessor.Builder()
                      .add(new ResizeWithCropOrPadOp(cropSize, cropSize))

                     .add(new ResizeOp(imageSizeX, imageSizeY, ResizeMethod.NEAREST_NEIGHBOR));
                     // .add(new ImageUtils.toGrayscale());
      //float [] imgData = ImageUtils.normalizeGray(newImg);
     // int [] shape = {1,imageSizeY, imageSizeX,1};
     // inputImageBuffer.load (imgData, shape);

      return  imageProcessor.process(inputImageBuffer);

      //return imageProcessor.process(inputImageBuffer);
    }*/
        else {
            imageProcessor =
                    new ImageProcessor.Builder()
                            .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                            .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                          //  .add(new Rot90Op(numRotation))
                            .add(getPreprocessNormalizeOp())
                            .build();
            return imageProcessor.process(inputImageBuffer);
        }

    }

    /** Gets the top-k results. */

    private static List<Recognition> getTopKProbability(Map<String, Float> labelProb) {
        // Find the best classifications.

        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
            pq.add(new Recognition("" + entry.getKey(), entry.getKey(), entry.getValue(), null));
            System.out.println(entry.getValue());
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        // System.out.println(pq.size());
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        return recognitions;
    }

    /** Gets the top-k results. */
    private static List<Recognition> getTopKProbabilityWithLocation(Map<String, Object[] > labelProb) {
        // Find the best classifications.

        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (Map.Entry<String, Object[]> entry : labelProb.entrySet()) {
            Object [] values = entry.getValue();
            double conf = (double) values[0];
            pq.add(new Recognition("" + entry.getKey(), entry.getKey(), (float) conf, (RectF) values[1]));
            System.out.println(entry.getValue());
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        // System.out.println(pq.size());
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        return recognitions;
    }


    /** Gets the name of the model file stored in Assets. */
    protected abstract String getModelPath();

    /** Gets the name of the label file stored in Assets. */
    protected abstract String getLabelPath();

    /** Gets the TensorOperator to nomalize the input image in preprocessing. */
    protected abstract TensorOperator getPreprocessNormalizeOp();

    /**
     * Gets the TensorOperator to dequantize the output probability in post processing.
     *
     * <p>For quantized model, we need de-quantize the prediction with NormalizeOp (as they are all
     * essentially linear transformation). For float model, de-quantize is not required. But to
     * uniform the API, de-quantize is added to float model too. Mean and std are set to 0.0f and
     * 1.0f, respectively.
     */
    protected abstract TensorOperator getPostprocessNormalizeOp();


}
