package com.mida.intelligentathlete;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.app.Activity;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothManager;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Parcelable;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.mbientlab.metawear.Data;
import com.mbientlab.metawear.DataProducer;
import com.mbientlab.metawear.MetaWearBoard;
import com.mbientlab.metawear.Route;
import com.mbientlab.metawear.Subscriber;
import com.mbientlab.metawear.android.BtleService;
import com.mbientlab.metawear.builder.RouteBuilder;
import com.mbientlab.metawear.builder.RouteComponent;
import com.mbientlab.metawear.data.Acceleration;
import com.mbientlab.metawear.data.AngularVelocity;
import com.mbientlab.metawear.module.Accelerometer;
import com.mbientlab.metawear.module.Gyro;
import com.mbientlab.metawear.module.Logging;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;

import bolts.Continuation;
import bolts.Task;

public class MainActivity extends AppCompatActivity implements ServiceConnection {
    private BtleService.LocalBinder serviceBinder;
    private final String wrist = "D5:90:D7:D3:5D:36", ankle = "D3:13:D6:52:47:23",
    accHeader = "epoch (ms),time (01:00),elapsed (s),x-axis (g),y-axis (g),z-axis (g)",
    gyrHeader = "epoch (ms),time (01:00),elapsed (s),x-axis (deg/s),y-axis (deg/s),z-axis (deg/s)";
    private MetaWearBoard boardWrist, boardAnkle;
    private Accelerometer accelerometerWrist, accelerometerAnkle;
    private Gyro gyroscopeWrist, gyroscopeAnkle;
    private Logging loggingWrist, loggingAnkle;
    private StringBuilder wristAcc = null, wristGyr = null, ankleAcc = null, ankleGyr = null;
    private boolean routed = true;
    private Intent homepage;
    RecyclerView recyclerView;
    RecyclerAdapter adapter;
    String data;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (Build.VERSION.SDK_INT >= 21) {
            Window window = this.getWindow();
            window.addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS);
            window.clearFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_STATUS);
            window.setStatusBarColor(this.getResources().getColor(R.color.grey));
        }

        // Bind the service when the activity is created
        getApplicationContext().bindService(new Intent(this, BtleService.class),
                this, Context.BIND_AUTO_CREATE);

        if (! Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }

        findViewById(R.id.start).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v){
                startAllSensors();
                findViewById(R.id.start).setEnabled(false);
                findViewById(R.id.stop).setEnabled(true);
                ((TextView) findViewById(R.id.text_instructions)).setText("Press STOP to download the data");
            }
        });

        findViewById((R.id.stop)).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                findViewById(R.id.stop).setEnabled(false);
                stopAllSensors();

                // Create the string builders of the data with their headers
                wristAcc = new StringBuilder();
                ankleAcc = new StringBuilder();
                wristGyr = new StringBuilder();
                ankleGyr = new StringBuilder();
                wristAcc.append(accHeader);
                ankleAcc.append(accHeader);
                wristGyr.append(gyrHeader);
                ankleGyr.append(gyrHeader);

                // Download the data from the loggers
                ((TextView) findViewById(R.id.text_instructions)).setText("Downloading the data...");

                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        Task<Void> downloadWrist = downloadData(loggingWrist,findViewById(R.id.progressWrist)).continueWith(new Continuation<Void, Void>() {
                            @Override
                            public Void then(Task<Void> task) throws Exception {
                                if(!task.isFaulted()) {
                                    // Delete pre-existing files
                                    deleteFile("wristAcc.csv");
                                    deleteFile("wristGyr.csv");
                                    // Create the csv files from the string builders
                                    createFile("wristAcc.csv", wristAcc);
                                    createFile("wristGyr.csv", wristGyr);
                                }
                                return null;
                            }
                        });
                        Task<Void> downloadAnkle = downloadData(loggingAnkle,findViewById(R.id.progressAnkle)).continueWith(new Continuation<Void, Void>() {
                            @Override
                            public Void then(Task<Void> task) throws Exception {
                                if(!task.isFaulted()){
                                    // Delete pre-existing files
                                    deleteFile("ankleAcc.csv");
                                    deleteFile("ankleGyr.csv");
                                    // Create the csv files from the string builders
                                    createFile("ankleAcc.csv",ankleAcc);
                                    createFile("ankleGyr.csv",ankleGyr);
                                }
                                return null;
                            }
                        });
                        ArrayList<Task<Void>> list = new ArrayList<Task<Void>>();
                        list.add(downloadWrist);
                        list.add(downloadAnkle);
                        Task.whenAll(list).continueWith(new Continuation<Void, Void>() {
                            @Override
                            public Void then(Task<Void> task) throws Exception {
                                if(!task.isFaulted()){
                                    // Updating the UI
                                    update_text("Download completed! Press ANALYZE to analyze the workout");
                                    enable_button(findViewById(R.id.start));
                                    enable_button(findViewById(R.id.print));
                                }
                                else{
                                    update_text("Download failed! Please try again");
                                }
                                return null;
                            }
                        });
                    }
                }).start();
            }
        });

        findViewById(R.id.reset).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                boardWrist.tearDown();
                boardAnkle.tearDown();

                // Delete the files
                deleteFile("wristAcc.csv");
                deleteFile("wristGyr.csv");
                deleteFile("ankleAcc.csv");
                deleteFile("ankleGyr.csv");
            }
        });

        findViewById(R.id.print).setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v) {
                ((TextView) findViewById(R.id.text_instructions)).setText("Analyzing the workout...");
                findViewById(R.id.progressWrist).setVisibility(View.GONE);
                findViewById(R.id.progressAnkle).setVisibility(View.GONE);
                findViewById(R.id.progressAnalysis).setVisibility(View.VISIBLE);
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        Python py = Python.getInstance();
                        PyObject classifier = py.getModule("classifier");
                        PyObject XGB = classifier.callAttr("XGB");
                        data = XGB.toString();
                    }
                }).start();
                findViewById(R.id.progressAnalysis).setVisibility(View.GONE);
                ((TextView) findViewById(R.id.text_instructions)).setText("ACTIVITIES");
                Log.i("IntAt",data);
                display(data);
            }
        });

        findViewById(R.id.download).setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v) {
                Context context = getApplicationContext();
                File wristAccFile = retrieveFile("wristAcc.csv"),
                ankleAccFile = retrieveFile("ankleAcc.csv"),
                wristGyrFile = retrieveFile("wristGyr.csv"),
                ankleGyrFile = retrieveFile("ankleGyr.csv");
                Uri wristAccPath = getURI(context,wristAccFile),
                ankleAccPath = getURI(context,ankleAccFile),
                wristGyrPath = getURI(context,wristGyrFile),
                ankleGyrPath = getURI(context,ankleGyrFile);
                ArrayList<Uri> files = new ArrayList<Uri>();
                files.add(wristAccPath);
                files.add(ankleAccPath);
                files.add(wristGyrPath);
                files.add(ankleGyrPath);
                Intent fileIntent = new Intent(Intent.ACTION_SEND_MULTIPLE);
                fileIntent.setType("text/csv");
                fileIntent.putExtra(Intent.EXTRA_SUBJECT,"Data Collected");
                fileIntent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
                fileIntent.putParcelableArrayListExtra(Intent.EXTRA_STREAM,files);
                startActivity(Intent.createChooser(fileIntent,"Send mail"));
            }
        });
    }

    @Override
    public void onDestroy() {
        super.onDestroy();

        // Tear the boards down
        boardWrist.tearDown();
        boardAnkle.tearDown();

        // Unbind the service when the activity is destroyed
        getApplicationContext().unbindService(this);
    }

    @Override
    public void onStop(){
        super.onStop();

        // Tear the boards down
        boardWrist.tearDown();
        boardAnkle.tearDown();
    }

    @Override
    public void onServiceConnected(ComponentName name, IBinder service) {
        // Typecast the binder to the service's LocalBinder class
        serviceBinder = (BtleService.LocalBinder) service;
        new Thread(new Runnable() {
            @Override
            public void run() {
                retrieveBoard(wrist, ankle);
            }
        }).start();
    }

    @Override
    public void onServiceDisconnected(ComponentName name) {
        // Tear the boards down
        boardWrist.tearDown();
        boardAnkle.tearDown();
    }

    private void retrieveBoard(String wrist, String ankle) {
        final BluetoothManager btManager =
                (BluetoothManager) getSystemService(Context.BLUETOOTH_SERVICE);
        final BluetoothDevice remoteDeviceWrist =
                btManager.getAdapter().getRemoteDevice(wrist);
        final BluetoothDevice remoteDeviceAnkle =
                btManager.getAdapter().getRemoteDevice(ankle);

        // Create a MetaWear board object for the Bluetooth Device
        boardWrist = serviceBinder.getMetaWearBoard(remoteDeviceWrist);
        boardAnkle = serviceBinder.getMetaWearBoard(remoteDeviceAnkle);

        // Establish BT connection
        Task<Void> connectedWrist = connectBoard(boardWrist,((CheckBox) findViewById(R.id.checkWrist)));
        Task<Void> connectedAnkle = connectBoard(boardAnkle,((CheckBox) findViewById(R.id.checkAnkle)));
        ArrayList<Task<Void>> list = new ArrayList<Task<Void>>();
        list.add(connectedWrist);
        list.add(connectedAnkle);
        Task.whenAll(list).continueWith(new Continuation<Void, Void>() {
            @Override
            public Void then(Task<Void> task) throws Exception {
                if(!task.isFaulted()){
                    // Retrieve accelerometers and gyroscopes
                    accelerometerWrist = retrieveAccelerometer(boardWrist);
                    accelerometerAnkle = retrieveAccelerometer(boardAnkle);
                    gyroscopeWrist = retrieveGyroscope(boardWrist);
                    gyroscopeAnkle = retrieveGyroscope(boardAnkle);

                    // Create data routes
                    Task<Void> routedAccWrist = createRoute(accelerometerWrist.acceleration(),WRIST_ACCELEROMETER_SUBSCRIBER);
                    Task<Void> routedAccAnkle = createRoute(accelerometerAnkle.acceleration(),ANKLE_ACCELEROMETER_SUBSCRIBER);
                    Task<Void> routedGyrWrist = createRoute(gyroscopeWrist.angularVelocity(),WRIST_GYROSCOPE_SUBSCRIBER);
                    Task<Void> routedGyrAnkle = createRoute(gyroscopeAnkle.angularVelocity(),ANKLE_GYROSCOPE_SUBSCRIBER);
                    ArrayList<Task<Void>> list = new ArrayList<Task<Void>>();
                    list.add(routedAccWrist);
                    list.add(routedAccAnkle);
                    list.add(routedGyrWrist);
                    list.add(routedGyrAnkle);
                    Task.whenAll(list).continueWith(new Continuation<Void, Void>() {
                        @Override
                        public Void then(Task<Void> task) throws Exception {
                            if(routed){
                                enable_button(findViewById(R.id.start));
                                update_text("All sensors configured correctly! Press START and begin the workout");
                                findViewById(R.id.progressAnalysis).post(new Runnable() {
                                    @Override
                                    public void run() {
                                        findViewById(R.id.progressAnalysis).setVisibility(View.INVISIBLE);
                                    }
                                });

                                // Create logging interfaces
                                loggingWrist = boardWrist.getModule(Logging.class);
                                loggingAnkle = boardAnkle.getModule(Logging.class);
                            }
                            else{
                                update_text("Failed to configure one or more sensors! Try restarting the app");
                                findViewById(R.id.progressAnalysis).post(new Runnable() {
                                    @Override
                                    public void run() {
                                        findViewById(R.id.progressAnalysis).setVisibility(View.INVISIBLE);
                                    }
                                });
                            }
                            return null;
                        }
                    });
                }
                else{
                    update_text("Failed to retrieve one or more devices! Try restarting the app");
                    findViewById(R.id.progressAnalysis).post(new Runnable() {
                        @Override
                        public void run() {
                            findViewById(R.id.progressAnalysis).setVisibility(View.INVISIBLE);
                        }
                    });
                }
                return null;
            }
        });
    }

    private Task<Void> connectBoard(MetaWearBoard board, CheckBox check){
        return board.connectWithRetryAsync(3).continueWith(new Continuation<Void, Void>() {
            @Override
            public Void then(Task<Void> task) throws Exception {
                if (task.isFaulted()) {
                    Log.i("IntAt", "Failed to connect to " + " " + board.getMacAddress());
                } else {
                    Log.i("IntAt", "Connected to " + " " + board.getMacAddress());
                    check.post(new Runnable() {
                        @Override
                        public void run() {
                            check.setChecked(true);
                        }
                    });
                }
                return null;
            }
        });
    }

    private Accelerometer retrieveAccelerometer(MetaWearBoard board){
        Accelerometer acc;
        acc = board.getModule(Accelerometer.class);
        acc.configure().odr(100f).commit(); // Set sampling frequency to 100Hz
        return acc;
    }

    private Gyro retrieveGyroscope(MetaWearBoard board){
        Gyro gyr;
        gyr = board.getModule(Gyro.class);
        gyr.configure().odr(Gyro.OutputDataRate.ODR_100_HZ).commit();
        return gyr;
    }

    private Task<Void> downloadData(Logging log, ProgressBar progress){
        progress.post(new Runnable() {
            @Override
            public void run() {
                progress.setVisibility(View.VISIBLE);
            }
        });

        return log.downloadAsync(100, (nEntriesLeft, totalEntries) -> {
            Log.i("IntAt", "Progress Update = " + (totalEntries - nEntriesLeft) + "/" + totalEntries);
            progress.post(new Runnable() {
                @Override
                public void run() {
                    progress.setMax((int) totalEntries);
                    progress.setProgress((int) totalEntries - (int) nEntriesLeft);
                }
            });
        }).continueWith(new Continuation<Void, Void>() {
            @Override
            public Void then(Task<Void> task) throws Exception {
                if(task.isFaulted()){
                    Log.i("IntAt","Log download failed!");
                } else{
                    Log.i("IntAt","Log download succeeded!");
                }
                log.clearEntries();
                return null;
            }
        });
    }

    private Task<Void> createRoute(DataProducer producer, Subscriber sub){
        return producer.addRouteAsync(new RouteBuilder() {
            @Override
            public void configure(RouteComponent source) {
                source.log(sub);
            }
        }).continueWith(new Continuation<Route, Void>() {
            @Override
            public Void then(Task<Route> task) throws Exception {
                if (task.isFaulted()) {
                    Log.i("IntAt", "Failed to route" + " " + producer.toString());
                    routed = false;
                } else {
                    Log.i("IntAt", producer.toString() + " " + "routed!");
                }
                return null;
            }
        });
    }

    private void startAllSensors(){
        loggingWrist.start(true);
        loggingAnkle.start(true);
        accelerometerWrist.acceleration().start();
        accelerometerAnkle.acceleration().start();
        gyroscopeWrist.angularVelocity().start();
        gyroscopeAnkle.angularVelocity().start();
        accelerometerWrist.start();
        accelerometerAnkle.start();
        gyroscopeWrist.start();
        gyroscopeAnkle.start();
    }

    private void stopAllSensors(){
        accelerometerWrist.stop();
        accelerometerAnkle.stop();
        gyroscopeWrist.stop();
        gyroscopeAnkle.stop();
        accelerometerWrist.acceleration().stop();
        accelerometerAnkle.acceleration().stop();
        gyroscopeWrist.angularVelocity().stop();
        gyroscopeAnkle.angularVelocity().stop();
        loggingWrist.stop();
        loggingAnkle.stop();
    }

    private void createFile(String name, StringBuilder body){
        try {
            FileOutputStream file = openFileOutput(name,Context.MODE_PRIVATE);
            file.write(body.toString().getBytes());
            file.close();
            Log.i("IntAt", name + " created!");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private File retrieveFile(String name){
        return new File(getFilesDir(),name);
    }

    private Uri getURI(Context context, File fileLocation){
        return FileProvider.getUriForFile(context,"com.mida.intelligentathlete.fileprovider",fileLocation);
    }

    // Subscribers
    private Subscriber WRIST_ACCELEROMETER_SUBSCRIBER = (Data data, Object... env) -> {
        String value = "\n" + data.timestamp().getTimeInMillis() + "," +
                data.formattedTimestamp() + "," +
                String.valueOf(0) + "," +
                data.value(Acceleration.class).x() + "," +
                data.value(Acceleration.class).y() + "," +
                data.value(Acceleration.class).z();
        Log.i("IntAt",value);
        wristAcc.append(value);
    };

    private Subscriber ANKLE_ACCELEROMETER_SUBSCRIBER = (Data data, Object... env) -> {
        String value = "\n" + data.timestamp().getTimeInMillis() + "," +
                data.formattedTimestamp() + "," +
                String.valueOf(0) + "," +
                data.value(Acceleration.class).x() + "," +
                data.value(Acceleration.class).y() + "," +
                data.value(Acceleration.class).z();
        Log.i("IntAt",value);
        ankleAcc.append(value);
    };

    private Subscriber WRIST_GYROSCOPE_SUBSCRIBER = (Data data, Object... env) -> {
        String value = "\n" + data.timestamp().getTimeInMillis() + "," +
                data.formattedTimestamp() + "," +
                String.valueOf(0) + "," +
                data.value(AngularVelocity.class).x() + "," +
                data.value(AngularVelocity.class).y() + "," +
                data.value(AngularVelocity.class).z();
        Log.i("IntAt",value);
        wristGyr.append(value);
    };

    private Subscriber ANKLE_GYROSCOPE_SUBSCRIBER = (Data data, Object... env) -> {
        String value = "\n" + data.timestamp().getTimeInMillis() + "," +
                data.formattedTimestamp() + "," +
                String.valueOf(0) + "," +
                data.value(AngularVelocity.class).x() + "," +
                data.value(AngularVelocity.class).y() + "," +
                data.value(AngularVelocity.class).z();
        Log.i("IntAt",value);
        ankleGyr.append(value);
    };

    // Display the data
    private void display(String data){
        String lines[] = data.split("\\r?\\n");
        for (int i = 0; i < lines.length; i++) {
            String elements[] = lines[i].split(",");
            lines[i] = elements[0] + ": " + elements[1] + " - " + elements[2];
        }

        recyclerView = findViewById(R.id.recycler);
        recyclerView.setVisibility(View.VISIBLE);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        adapter = new RecyclerAdapter(this,lines);
        recyclerView.setAdapter(adapter);
    }

    // Helper functions: update the UI from a thread
    private void update_text(String to_change){
        findViewById(R.id.text_instructions).post(new Runnable() {
            @Override
            public void run() {
                ((TextView) findViewById(R.id.text_instructions)).setText(to_change);
            }
        });
    }

    private void enable_button(Button button){
        button.post(new Runnable() {
            @Override
            public void run() {
                button.setEnabled(true);
            }
        });
    }

    private void disable_button(Button button){
        button.post(new Runnable() {
            @Override
            public void run() {
                button.setEnabled(false);
            }
        });
    }
}