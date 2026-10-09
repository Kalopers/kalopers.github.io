'use strict';

function symbolCount(value) {
  return String(value || '').length;
}

function formatCount(value) {
  if (value > 9999) return `${Math.round(value / 1000)}k`;
  if (value > 999) return `${Math.round(value / 100) / 10}k`;
  return value;
}

function totalSymbols(site) {
  return (site.posts || []).reduce((total, post) => total + symbolCount(post.content), 0);
}

hexo.extend.helper.register('symbolsCount', value => formatCount(symbolCount(value)));
hexo.extend.helper.register('wordcount', value => formatCount(symbolCount(value)));
hexo.extend.helper.register('totalcount', site => formatCount(totalSymbols(site)));
hexo.extend.helper.register('symbolsCountTotal', site => formatCount(totalSymbols(site)));
hexo.extend.helper.register('symbolsTime', value => Math.max(1, Math.round(symbolCount(value) / 275)) + ' mins.');
hexo.extend.helper.register('min2read', value => Math.max(1, Math.round(symbolCount(value) / 275)) + ' mins.');
hexo.extend.helper.register('symbolsTimeTotal', site => Math.max(1, Math.round(totalSymbols(site) / 275)) + ' mins.');
