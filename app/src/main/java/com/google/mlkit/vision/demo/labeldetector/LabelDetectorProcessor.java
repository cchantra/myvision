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

package com.google.mlkit.vision.demo.labeldetector;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.demo.BitmapUtils;
import com.google.mlkit.vision.demo.DetectionMode;
import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.VisionProcessorBase;
import com.google.mlkit.vision.demo.Constant;
import com.google.mlkit.vision.label.ImageLabel;
import com.google.mlkit.vision.label.ImageLabeler;
import com.google.mlkit.vision.label.ImageLabelerOptionsBase;
import com.google.mlkit.vision.label.ImageLabeling;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;

/**
 * Custom InputImage Classifier Demo.
 */



public class LabelDetectorProcessor extends VisionProcessorBase<List<ImageLabel>> {

    private static final String TAG = "LabelDetectorProcessor";

    private final ImageLabeler imageLabeler;


    private String testStr = "...Test";
    private int model_option   = Constant.AGE_OPTION;



    public LabelDetectorProcessor(Context context, ImageLabelerOptionsBase options) {
        super(context);
        imageLabeler = ImageLabeling.getClient(options);
    }

    public LabelDetectorProcessor(Context context, ImageLabelerOptionsBase options, String str) {
        super(context);
        imageLabeler = ImageLabeling.getClient(options);
        testStr = str;
    }

    public LabelDetectorProcessor(Context context, ImageLabelerOptionsBase options, int opt_model) {
        super(context);
        imageLabeler = ImageLabeling.getClient(options);
        model_option = opt_model;
    }


    @Override
    public void stop() {
        super.stop();
        try {
            imageLabeler.close();
        } catch (IOException e) {
            Log.e(TAG, "Exception thrown while trying to close ImageLabelerClient: " + e);
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    @Override
    protected Task<List<ImageLabel>> detectInImage(InputImage image) {
        if (model_option == Constant.AGE_OPTION) {
            // preprocess image
            ByteBuffer imageByte = image.getByteBuffer();

             BitmapUtils.getBitmap(imageByte);

        }
        else
            return imageLabeler.process(image);

    }

    @Override
    protected void onSuccess(
            @NonNull List<ImageLabel> labels, @NonNull GraphicOverlay graphicOverlay) {
        graphicOverlay.add(new LabelGraphic(graphicOverlay, labels, testStr));
        logExtrasForTesting(labels);
    }

    private static void logExtrasForTesting(List<ImageLabel> labels) {
        if (labels == null) {
            Log.v(MANUAL_TESTING_LOG, "No labels detected");
        } else {
            for (ImageLabel label : labels) {
                Log.v(
                        MANUAL_TESTING_LOG,
                        String.format("Label %s, confidence %f", label.getText(), label.getConfidence()));
            }
        }
    }

    @Override
    protected void onFailure(@NonNull Exception e) {
        Log.w(TAG, "Label detection failed." + e);
    }
}

