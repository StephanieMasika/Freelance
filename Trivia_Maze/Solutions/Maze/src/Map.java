import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.util.*;

public class Map
{
    private Scanner m;
    private String Map[] = new String[14];
    private Image grass;
    public Image wall;

    public Map()
    {
        ImageIcon img = new ImageIcon("E:\\Personal Home\\Work\\Freelancing\\New\\Done\\Trivia Maze\\grass.jpeg");
        grass = img.getImage();

        img = new ImageIcon("E:\\Personal Home\\Work\\Freelancing\\New\\Done\\Trivia Maze\\wall.jpeg");
        wall = img.getImage();

        openFile();
        readFile();
        closeFile();
    }

    public Image getGrass()
    {
        return grass;
    }

    public Image getWall()
    {
        return wall;
    }

    public String getMap(int x, int y)
    {
        String index = Map[y].substring(x, x + 1);

        return index;
    }

    public void openFile()
    {
        try
        {
            m = new Scanner(new File("E:\\Personal Home\\Work\\Freelancing\\New\\Done\\Trivia Maze\\Map.txt"));
        }
        catch (Exception e)
        {
            System.out.println("error loading map");
        }
    }

    public void readFile()
    {
        while (m.hasNext())
        {
            for (int i = 0; i < 14; i++)
            {
                Map[i] = m.next();
            }
        }
    }

    public void closeFile()
    {
        m.close();
    }
}
