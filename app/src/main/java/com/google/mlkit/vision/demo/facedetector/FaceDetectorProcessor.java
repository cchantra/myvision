/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mlkit.vision.demo.facedetector;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.PointF;
import android.graphics.RectF;
import android.os.SystemClock;
import android.util.Log;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.demo.BitmapUtils;
import com.google.mlkit.vision.demo.Constant;
import com.google.mlkit.vision.demo.FaceAPIUtil;
import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.LivePreviewActivity;
import com.google.mlkit.vision.demo.Logger;
import com.google.mlkit.vision.demo.VisionProcessorBase;
import com.google.mlkit.vision.demo.constantURL;
import com.google.mlkit.vision.demo.tflite.Classifier;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.mlkit.vision.face.FaceLandmark;

import org.json.JSONArray;
import org.json.JSONObject;

import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * Face Detector Demo.
 */
public class FaceDetectorProcessor extends VisionProcessorBase<List<Face>> {

    private static final String TAG = "FaceDetectorProcessor";
    private static final Logger LOGGER = new Logger();
    private static final int MAX_RESULTS = 3;

    private final FaceDetector detector;

    private  int sensorOrientation;

    private int mode;
    private Context context;
    private Classifier classifier;


    public FaceDetectorProcessor(Context context, int mode,  Classifier classifier) {
        this(
                context,
                new FaceDetectorOptions.Builder()
                        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                        .enableTracking()
                        .build(), mode,   classifier);

      this.mode = mode;
      this.context = context;

      this.classifier = classifier;



    }

    public FaceDetectorProcessor(Context context, FaceDetectorOptions options, int mode,   Classifier classifier ) {
        super(context);
        Log.v(MANUAL_TESTING_LOG, "Face detector options: " + options);
        detector = FaceDetection.getClient(options);
        this.mode = mode;
        this.context = context;

        this.classifier = classifier;

    }

   /* public FaceDetectorProcessor(Context context, FaceDetectorOptions build) {
        super(context);
    }*/

    @Override
    public void stop() {
        super.stop();
        detector.close();
    }

    @Override
    protected Task<List<Face>> detectInImage(InputImage image) {


        return detector.process(image);


    }

    @Override
    protected void onSuccess(@NonNull List<Face> faces, @NonNull Bitmap bitmap, @NonNull GraphicOverlay graphicOverlay) throws MalformedURLException {


        if (mode == Constant.TEST_JSON) {
            final long startTime = SystemClock.uptimeMillis();
            Bitmap [] bma = {bitmap};
            List<List<Classifier.Recognition>> results =  new ArrayList<List<Classifier.Recognition>>();

            //classifier.recognizeImageAll(bitmap, results);

            FaceAPIUtil asyncTask = new FaceAPIUtil(new URL(constantURL.FaceURL), output -> {
                //Here you will receive the result fired from async class
                //of onPostExecute(result) method.

                Log.e("output", output);

                // parse : {'emotion': [['sad'], ['neutral']], 'detail': [[[1, 2, 3, 4], [0.56]], [[1, 2, 3, 5], [0.44]]]}
                JSONObject json = new JSONObject(output);
                JSONArray emotionArray = (JSONArray) json.get("emotion");
                JSONArray detailArray = (JSONArray) json.get("detail");

                for (int i=0; i < emotionArray.length(); i++) { // for each face
                    System.out.println(emotionArray.get(i));
                    System.out.println(detailArray.get(i));

                    Map<String, Object []> labeledProbability =  new HashMap<String, Object[]>();


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
                    //recogListAll.add( getTopKProbabilityWithLocation(labeledProbability));
                    results.add( getTopKProbabilityWithLocation(labeledProbability));
                }


                System.out.println("added "+results.size());
                LOGGER.v("Detect: %s", results);
                System.out.println(  " xxx" + results);

                //for (Face face : faces) {
                int i=0;
                for (Face face : faces) {
                    if (results.size() == 0)
                        break;
                    List<Classifier.Recognition> result = results.get(i);
                    i++;
                    graphicOverlay.add(new FaceGraphic(graphicOverlay, face, i, result )); // must draw itself

                    logExtrasForTesting(face);


                    LivePreviewActivity imageView = (LivePreviewActivity) context;
                    imageView.showObjectID(i);
                    imageView.showResultsInBottomSheet(result);

                }

            });

            asyncTask.execute(bma);




        }
        else {
            int i =0;
            for (Face face : faces) {

                i++;

                graphicOverlay.add(new FaceGraphic(graphicOverlay, face, i, null));


                final long startTime = SystemClock.uptimeMillis();
                Bitmap croppedImage = BitmapUtils.cropBitmap(bitmap, face.getBoundingBox());
                final List<Classifier.Recognition> results =
                        classifier.recognizeImage(croppedImage);
                long lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                LOGGER.v("Detect: %s", results);
                System.out.println(i + " xxx" + results);

                logExtrasForTesting(face);

                LivePreviewActivity imageView = (LivePreviewActivity) context;
                imageView.showObjectID(i);
                imageView.showResultsInBottomSheet(results);


            }
        }
    }



    private static List<Classifier.Recognition> getTopKProbabilityWithLocation(Map<String, Object[] > labelProb) {
        // Find the best classifications.

        PriorityQueue<Classifier.Recognition> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<Classifier.Recognition>() {
                            @Override
                            public int compare(Classifier.Recognition lhs, Classifier.Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (Map.Entry<String, Object[]> entry : labelProb.entrySet()) {
            Object [] values = entry.getValue();
            double conf = (double) values[0];
            pq.add(new Classifier.Recognition("" + entry.getKey(), entry.getKey(), (float) conf, (RectF) values[1]));
            System.out.println(entry.getValue());
        }

        final ArrayList<Classifier.Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        // System.out.println(pq.size());
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        return recognitions;
    }
    private static void logExtrasForTesting(Face face) {
        if (face != null) {
            Log.v(MANUAL_TESTING_LOG, "face bounding box: " + face.getBoundingBox().flattenToString());
            Log.v(MANUAL_TESTING_LOG, "face Euler Angle X: " + face.getHeadEulerAngleX());
            Log.v(MANUAL_TESTING_LOG, "face Euler Angle Y: " + face.getHeadEulerAngleY());
            Log.v(MANUAL_TESTING_LOG, "face Euler Angle Z: " + face.getHeadEulerAngleZ());

            // All landmarks
            int[] landMarkTypes =
                    new int[]{
                            FaceLandmark.MOUTH_BOTTOM,
                            FaceLandmark.MOUTH_RIGHT,
                            FaceLandmark.MOUTH_LEFT,
                            FaceLandmark.RIGHT_EYE,
                            FaceLandmark.LEFT_EYE,
                            FaceLandmark.RIGHT_EAR,
                            FaceLandmark.LEFT_EAR,
                            FaceLandmark.RIGHT_CHEEK,
                            FaceLandmark.LEFT_CHEEK,
                            FaceLandmark.NOSE_BASE
                    };
            String[] landMarkTypesStrings =
                    new String[]{
                            "MOUTH_BOTTOM",
                            "MOUTH_RIGHT",
                            "MOUTH_LEFT",
                            "RIGHT_EYE",
                            "LEFT_EYE",
                            "RIGHT_EAR",
                            "LEFT_EAR",
                            "RIGHT_CHEEK",
                            "LEFT_CHEEK",
                            "NOSE_BASE"
                    };
            for (int i = 0; i < landMarkTypes.length; i++) {
                FaceLandmark landmark = face.getLandmark(landMarkTypes[i]);
                if (landmark == null) {
                    Log.v(
                            MANUAL_TESTING_LOG,
                            "No landmark of type: " + landMarkTypesStrings[i] + " has been detected");
                } else {
                    PointF landmarkPosition = landmark.getPosition();
                    String landmarkPositionStr =
                            String.format(Locale.US, "x: %f , y: %f", landmarkPosition.x, landmarkPosition.y);
                    Log.v(
                            MANUAL_TESTING_LOG,
                            "Position for face landmark: "
                                    + landMarkTypesStrings[i]
                                    + " is :"
                                    + landmarkPositionStr);
                }
            }
            Log.v(
                    MANUAL_TESTING_LOG,
                    "face left eye open probability: " + face.getLeftEyeOpenProbability());
            Log.v(
                    MANUAL_TESTING_LOG,
                    "face right eye open probability: " + face.getRightEyeOpenProbability());
            Log.v(MANUAL_TESTING_LOG, "face smiling probability: " + face.getSmilingProbability());
            Log.v(MANUAL_TESTING_LOG, "face tracking id: " + face.getTrackingId());
        }
    }

    @Override
    protected void onFailure(@NonNull Exception e) {
        Log.e(TAG, "Face detection failed " + e);
    }
}
