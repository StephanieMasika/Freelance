import javax.swing.*;

public class Maze
{
    public Maze()
    {
        JFrame f = new JFrame();
        f.setTitle("Maze Trivia");
        f.add(new Board());
        f.setSize(500, 400);
        f.setLocationRelativeTo(null);
        f.setVisible(true);
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

}
