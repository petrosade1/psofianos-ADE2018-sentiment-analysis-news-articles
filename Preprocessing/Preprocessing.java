package cue.lang;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.text.Normalizer;
import java.util.Scanner;

import cue.lang.stop.StopWords;
// the code must be used with the other classes from https://github.com/jdf/cue.language
public class Preprocessing {
	public static boolean isNumeric(final String str) {
		try {
			double d = Double.parseDouble(str);
		} catch (NumberFormatException nfe) {
			return false;
		}
		return true;
	}

	public static String stripAccents(String s) {
		s = Normalizer.normalize(s, Normalizer.Form.NFD);
		s = s.replaceAll("[\\p{InCombiningDiacriticalMarks}]", "");
		return s;
	}

	public static void main(final String[] args) throws IOException {
		// TODO Auto-generated method stub
		File file = new File("nopre.txt");
		System.setOut(new PrintStream(new FileOutputStream("outputbinary.txt")));
		Scanner sc2 = null;
		try {
			sc2 = new Scanner(file);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		while (sc2.hasNextLine()) {
			Scanner s2 = new Scanner(sc2.nextLine());
			int count = 0;
			while (s2.hasNext()) {
				String s = s2.next();

				
				if (StopWords.Greek.isStopWord(s) && (count > 3)) { 
					continue;
				}
				
			    if ((s.contains("@")) || (s.contains(".com")) || (s.contains(".gr"))) { //
				
				continue;
				}
				if (BlockUtil.guessUnicodeBlock(s) != null) {
				if (!(BlockUtil.guessUnicodeBlock(s).toString().equals("GREEK")) && (count >
				3)) { 
				
				 continue;
				}
				}
				String result = s.replaceAll("[+.^:,'»«]", "");
				
				result = stripAccents(result);
				s = result.toUpperCase();

				System.out.print(s + " ");
				count++;

			}
			System.out.println();

		}

	}

}
