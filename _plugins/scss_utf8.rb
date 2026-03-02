# Fix encoding issue with jekyll-sass-converter on Ruby 3+ where files
# are read as US-ASCII but Sass requires UTF-8 input.
Jekyll::Converters::Scss.class_eval do
  alias_method :_orig_convert, :convert

  def convert(content)
    _orig_convert(content.encode("UTF-8"))
  end
end
