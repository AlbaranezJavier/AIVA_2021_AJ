package co.ortizol;

import java.io.*;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) throws IOException {
        File f = new File("ruta.txt"); // Creamos un objeto file
        String root = f.getAbsolutePath();
        String[] parts = root.split("\\" + "\\");

        String command1 = "";
        for (int i = 0; i < parts.length-2; ++i) {
            command1 += parts[i] + "/";
        }

        //Entrada por teclado
        Scanner scannerin = new Scanner (System.in);
        System.out.println ("¿Desea analizar una imagen de Train o de Test?");
        String keyboardin = "";
        keyboardin = scannerin.nextLine();
        System.out.println ("¿Cual es el índice de la imagen que desea anlizar?");
        keyboardin += "/" + scannerin.nextLine();

        //Comandos
        String command2 = "cd " + command1;
        command1 += "venv/Scripts/activate.bat";
        String command3 = "python calculate_imperfections.py -image=./Samples/" + keyboardin;

        Process p = Runtime.getRuntime().exec(command1 + " && " + command2 + " && " + command3); //"C:/Users/andre/Documents/GitHub/AIVA_2021_AJ/venv/Scripts/activate.bat && cd C:/Users/andre/Documents/GitHub/AIVA_2021_AJ && python calculate_imperfections.py");
        BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream()));
        String ret = in.readLine();
        System.out.println(ret);
    }
}
