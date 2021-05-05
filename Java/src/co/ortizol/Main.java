package co.ortizol;

import java.io.*;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) throws IOException {
        //Entrada por teclado
        Scanner scannerin = new Scanner (System.in);
        System.out.println ("Do you want to analyze a Train or Test image?");
        String keyboardin = "";
        keyboardin = scannerin.nextLine();
        System.out.println ("What is the index of the image you want to analyze?");
        keyboardin += "/" + scannerin.nextLine();

        String command1 = "python3 /tmp/AIVA_2021_AJ/calculate_imperfections.py -image=/tmp/AIVA_2021_AJ/Samples/" + keyboardin;

        Process p = Runtime.getRuntime().exec(command1); //"C:/Users/andre/Documents/GitHub/AIVA_2021_AJ/venv/Scripts/activate.bat && cd C:/Users/andre/Documents/GitHub/AIVA_2021_AJ && python calculate_imperfections.py");
        BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream()));
        String ret = in.readLine();
        System.out.println(ret);
    }
}
