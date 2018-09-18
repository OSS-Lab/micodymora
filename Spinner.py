def spinner(t, max_t, pattern=".oOo", box_size=15):
    bar = "".join(i <= t / max_t * box_size
                  and pattern[(i + t) % len(pattern)] or " "
                  for i in range(box_size))
    return "[{}] {}/{}".format(bar, t, max_t)
