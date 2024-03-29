﻿uses GraphABC;

var
  p: Picture;
  t: text;
  c: char;
  i, j: integer; // Объявление переменной j

begin
  p := Picture.Create('image.png');
  p.Draw(250, 250, 128, 128);
  writeln('Какой символ на картинке: ');
  read(c);

  assign(t, 'test.txt');
  t.Append();

  for i := 0 to p.Height - 1 do
  begin
    for j := 0 to p.Width - 1 do
      write(t, (1.0 - (p.GetPixel(j, i).R) / 255):0:3, ' ');
    writeln(t, '');
  end;

  writeln(t, c);
  t.Close();

  writeln('done!');
  window.Close();
end.