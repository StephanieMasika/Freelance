import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

public class Board extends JPanel implements ActionListener
{
    private Timer timer;
    private Map m;

    public Board()
    {
        m = new Map();
        timer = new Timer(25, this);
        timer.start();

    }

    public void actionPerformed(ActionEvent e)
    {
        repaint();
    }

    public void paint(Graphics g)
    {
        super.paint(g);

        for (int y = 0; y < 14; y++)
        {
            for (int x = 0; x < 14; x++)
            {
                if (m.getMap(x, y).equals("g"))
                {
                    g.drawImage(m.getGrass(), x * 32, y * 32, null);
                }

                if (m.getMap(x, y).equals("g"))
                {
                    g.drawImage(m.getWall(), x * 32, y * 32, null);
                }
            }
        }
    }

}
